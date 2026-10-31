"""Focused logic tests; fake-environment scores are never portfolio results."""

import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cartpole_corrected import experiment as e


CONFIG = json.loads(Path(e.__file__).with_name("config.json").read_text(encoding="utf-8"))


class FakeSpace:
    shape = (4,)
    n = 2
    high = np.array([4.8, np.inf, 0.418879, np.inf])


class FakeEnv:
    def __init__(self):
        self.observation_space = FakeSpace()
        self.action_space = FakeSpace()
        self.spec = type("Spec", (), {"id": "CartPole-v1", "max_episode_steps": 200})()
        self.reset_seeds = []
        self.steps = 0
        self.closed = False

    def reset(self, seed):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.array([0.01, 0, 0.01, 0], dtype=np.float32), {}

    def step(self, action):
        assert action in (0, 1)
        self.steps += 1
        return np.array([0.02, 0, 0.02, 0], dtype=np.float32), 1.0, self.steps == 2, False, {}

    def close(self):
        self.closed = True


class ExperimentTests(unittest.TestCase):
    def test_reward_matches_historical_formula_and_is_symmetric(self):
        positive = [1.0, 0, 0.1, 0]
        negative = [-1.0, 0, -0.1, 0]
        expected = 0.15 * (math.exp((4.8 - 1) / 4.8) - 0.999) + 0.85 * (math.exp((0.208 - 0.1) / 0.208) - 0.999)
        self.assertAlmostEqual(e.shaped_reward(positive, 4.8, 0.208, 0.15), expected)
        self.assertAlmostEqual(e.shaped_reward(negative, 4.8, 0.208, 0.15), expected)

    def test_discounted_returns(self):
        torch.testing.assert_close(e.discounted_returns([1, 2, 3], 0.5), torch.tensor([2.75, 3.5, 3.0]))

    def test_config_rejects_overlapping_seeds_and_broken_budget(self):
        bad = copy.deepcopy(CONFIG)
        bad["evaluation_seeds"] = [bad["training_seeds"][0] * 1000]
        with self.assertRaisesRegex(ValueError, "overlap"):
            e.validate_config(bad)
        bad = copy.deepcopy(CONFIG)
        bad["checkpoints"] = [0, 2000, 6000]
        with self.assertRaisesRegex(ValueError, "Checkpoints"):
            e.validate_config(bad)

    def test_evaluation_uses_original_reward_and_fixed_initial_states(self):
        env = FakeEnv()
        scores = e.evaluate(env, lambda _observation: 0, [900000, 900001])
        self.assertEqual(scores, [2.0, 2.0])
        self.assertEqual(env.reset_seeds, [900000, 900001])

    def test_greedy_evaluation_does_not_consume_torch_rng(self):
        torch.manual_seed(7)
        policy = e.Policy(8)
        state = torch.random.get_rng_state().clone()
        action = e.greedy_action(policy, [0, 0, 0, 0])
        self.assertIn(action, (0, 1))
        torch.testing.assert_close(torch.random.get_rng_state(), state)

    def test_equal_step_budget_checkpoint_scores_and_repeatability(self):
        config = copy.deepcopy(CONFIG)
        config.update(training_steps=4, checkpoints=[0, 2, 4], evaluation_seeds=[900000, 900001])
        created = []

        def factory(_config):
            env = FakeEnv()
            created.append(env)
            return env

        with patch.object(e, "environment", factory):
            original = e.train_condition(config, 11, "environment_reward")
            shaped = e.train_condition(config, 11, "shaped_reward")
            repeat = e.train_condition(config, 11, "environment_reward")
        for result in (original, shaped, repeat):
            self.assertEqual(result["training_steps"], 4)
            self.assertEqual([p["training_steps"] for p in result["checkpoints"]], [0, 2, 4])
            self.assertEqual([p["evaluation_scores"] for p in result["checkpoints"]], [[2.0, 2.0]] * 3)
        self.assertEqual(original, repeat)
        self.assertTrue(all(env.closed for env in created))
        self.assertEqual(created[0].reset_seeds, [11000, 11001])
        self.assertTrue(set(created[0].reset_seeds).isdisjoint(created[1].reset_seeds))

    def test_summary_uses_training_seed_as_unit(self):
        config = {"training_seeds": [1, 2], "checkpoints": [0, 10]}
        runs = []
        for seed, original, shaped in [(1, 10.0, 13.0), (2, 20.0, 15.0)]:
            for condition, score in [("environment_reward", original), ("shaped_reward", shaped)]:
                runs.append({"seed": seed, "condition": condition, "checkpoints": [
                    {"training_steps": 0, "mean_score": 5.0},
                    {"training_steps": 10, "mean_score": score},
                ]})
        summary = e.summarize(runs, [{"mean_score": 4.0}, {"mean_score": 6.0}], config)
        self.assertEqual(summary["environment_reward"][-1]["mean"], 15.0)
        self.assertEqual(summary["paired_difference_summary"]["mean"], -1.0)
        self.assertEqual([p["shaped_minus_original"] for p in summary["paired_final_differences"]], [3.0, -5.0])


if __name__ == "__main__":
    unittest.main()
