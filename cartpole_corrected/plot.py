"""Render the saved learning curve as a standalone SVG without plotting packages."""

import argparse
import json
from pathlib import Path


SERIES = (
    ("environment_reward", "#1864ab", "Original training reward"),
    ("shaped_reward", "#d9480f", "Shaped training reward"),
)


def svg(result: dict) -> str:
    config, summary = result["config"], result["summary"]
    left, top, width, height = 76, 75, 720, 350
    budget, ceiling = config["training_steps"], config["max_episode_steps"]
    x = lambda value: left + width * value / budget
    y = lambda value: top + height * (1 - value / ceiling)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" role="img" aria-label="CartPole original-reward learning curves">',
        '<rect width="900" height="520" fill="white"/>',
        '<text x="76" y="38" font-family="sans-serif" font-size="22" font-weight="bold">CartPole reward comparison</text>',
        '<text x="76" y="58" font-family="sans-serif" font-size="12">Original environment reward; mean ± 1 SD across five training seeds</text>',
    ]
    for tick in range(0, ceiling + 1, 50):
        yy = y(tick)
        parts.extend([
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + width}" y2="{yy:.1f}" stroke="#e9ecef"/>',
            f'<text x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>',
        ])
    for tick in config["checkpoints"]:
        parts.append(f'<text x="{x(tick):.1f}" y="{top + height + 21}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick}</text>')
    baseline = summary["random_baseline"]["mean"]
    parts.append(f'<line x1="{left}" y1="{y(baseline):.1f}" x2="{left + width}" y2="{y(baseline):.1f}" stroke="#868e96" stroke-dasharray="6 5"/>')
    for condition, color, _label in SERIES:
        points = summary[condition]
        coords = " ".join(f'{x(p["training_steps"]):.1f},{y(p["mean"]):.1f}' for p in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for point in points:
            xx, mean, sd = x(point["training_steps"]), point["mean"], point["sd_across_seeds"]
            low, high = y(max(0, mean - sd)), y(min(ceiling, mean + sd))
            parts.extend([
                f'<line x1="{xx:.1f}" y1="{low:.1f}" x2="{xx:.1f}" y2="{high:.1f}" stroke="{color}" stroke-width="1.5"/>',
                f'<circle cx="{xx:.1f}" cy="{y(mean):.1f}" r="4" fill="{color}"/>',
            ])
    parts.extend([
        f'<text x="{left + width / 2}" y="{top + height + 51}" text-anchor="middle" font-family="sans-serif" font-size="14">Training environment steps</text>',
        f'<text transform="translate(22 {top + height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Evaluation episode reward (0–{ceiling})</text>',
        f'<text x="80" y="498" font-family="sans-serif" font-size="12" fill="#868e96">Random baseline: {baseline:.1f}</text>',
    ])
    for index, (_condition, color, label) in enumerate(SERIES):
        parts.append(f'<text x="{300 + 230 * index}" y="498" font-family="sans-serif" font-size="12" fill="{color}">{label}</text>')
    return "\n".join(parts + ['</svg>']) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("learning_curve.svg"))
    args = parser.parse_args()
    args.output.write_text(svg(json.loads(args.input.read_text(encoding="utf-8"))), encoding="utf-8")
