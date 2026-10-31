# Analytics & ML Portfolio

Applied analysis across credit risk, time series, customer behaviour, text, and reinforcement learning. This repository is a collection of **separate notebook studies**, plus a small synthetic banking analytics prototype. Original notebooks retain their code and saved outputs. The corrected forecasting and CartPole studies have newly calculated results. The corrected credit-risk and churn references have synthetic software tests but **no real-data model results**. Other results below are historical saved output, not independently rerun.

**Start with the [technical blueprint](PROJECT_BLUEPRINT.md)** for data flow, mathematical foundations, evaluation, reproducibility notes, and limitations.

## Featured work

| Project | What the source implements | Evidence and reading path |
| --- | --- | --- |
| [Credit risk PD scorecard — corrected reference](credit_risk_corrected/README.md) | Separate dated-outcome contract, chronological cohorts, train-only WoE/IV, regularized logistic bad-loan model, explicit points-to-PD mapping, and validation diagnostics. | **Tested modelling framework, not a validated production PD model.** Synthetic software tests only; no reproduced LendingClub metrics. The [original preparation](<Capstone_NIBM_Credit Risk Modeling (PD Scorecard)/1_Data-Preparation.ipynb>) and [model/scorecard](<Capstone_NIBM_Credit Risk Modeling (PD Scorecard)/2_PD Model & Scorecard.ipynb>) remain historical. Their saved AUROC **0.6883**, Gini **0.3766**, KS **0.2790** are not a validated 12-month PD result; the exact source and dated event history are absent. |
| [Antidiabetic prescription forecasting — corrected](<Forecasting of Antidiabetic Drug Prescriptions - Corrected.ipynb>) | Dependency-free conditional SARIMA study with correct lag-12 differencing, training-only AIC selection, and three rolling 12-month comparisons against ordinary and seasonal naive baselines. | **Newly calculated:** 36-month MAPE **8.869%** for the selected restricted model versus **12.687%** seasonal naive and **14.545%** ordinary naive. See the [reproduction guide](forecasting/README.md) and [blueprint](PROJECT_BLUEPRINT.md#4-forecasting-antidiabetic-prescriptions). The [original notebook](<Forecasting of Antidiabetic Drug Prescriptions.ipynb>) and its 7.899% saved model result remain historical. |
| [Bank churn — corrected reference](churn_corrected/README.md) | Separate schema-checked, train-fitted logistic pipeline with a stratified train/validation/test split, a prevalence baseline, validation-selected model and threshold, and focused synthetic tests. | **Tested framework; no real-data metrics reproduced.** The [original neural-network notebook](BankChurnPrediction.ipynb) is preserved. Its saved SGD ROC-AUC **0.8533** and SMOTE recall **0.75** are historical outputs; repeated test use limits model comparisons. |
| [CartPole reward shaping — controlled study](cartpole_corrected/README.md) | A separate five-seed REINFORCE comparison with equal environment-interaction budgets, disjoint evaluation seeds and a random-action baseline; all 29 historical notebooks remain intact. | **Newly calculated at 8,000 training steps:** original-reward evaluation 85.56 ± 40.77 versus shaped-reward 97.48 ± 44.27 (mean ± SD across seeds). The paired difference is +11.92 ± 36.66; this does **not** establish a reliable shaping advantage. [Raw results](cartpole_corrected/results.json) and [experiment map](PROJECT_BLUEPRINT.md#6-cartpole-reinforcement-learning). |
| [Employee retention modelling](<Employee Retention & Performance Analytics.ipynb>) | Logistic regression, decision tree, random forest, grid search, and separate feature sets on an HR attrition dataset. | Saved logistic test recall for leavers **0.26**; a random-forest feature-set variant reports test F1 **0.894**. The notebook labels ROC-AUC computed from hard predictions as “AUC.” |

## More analyses

| Project | Implemented scope |
| --- | --- |
| [Household rural/urban classification](Socio_Economic_Household_Classification.ipynb) | Survey EDA, feature engineering, logistic/tree/ANN/forest/XGBoost variants, SMOTE, weighting, PCA, and voting. Saved output includes a tuned forest test accuracy of **0.8356**; feature selection and repeated test comparisons need caution. |
| [Airline tweet sentiment](<Twitter Sentiment Analysis with NLP.ipynb>) | Text cleaning, count and TF-IDF vectors, random forests, five-fold CV and test reports. Saved count-vector test accuracy is **0.77**; vectorizers were fitted before the split. |
| [Social media use and stress](1_OTHER%20PROJECTS/SocialMediavsStress.ipynb) | Correlation and OLS models with gender interaction for stress and anxiety scores; 99 observations appear in saved output. Source CSVs are absent. |
| [Mall customer segmentation](<1_OTHER PROJECTS/102_Customer Segmentation_K-Means Clustering.ipynb>) | Two-feature K-means, within-cluster sum of squares elbow plot, and five cluster visualisation. |
| [IMDb scraping and analysis](<1_OTHER PROJECTS/IMDb Movie Data – Web Scraping & Analysis.ipynb>) | Requests/BeautifulSoup extraction, cleaning, and descriptive plots for a saved 1,000-film result. Site content may change. |
| [Synthetic ALM engine](ALM_Project/README.md) | Python scripts generate simulated assets/liabilities and calculate repricing gaps, a +100 bp NII shock, and a simplified LCR. Saved reports are examples, not live bank results. |
| [Synthetic small-finance-bank data lab](sfb-risk-lab/data/README.md) | A small customer/loan/payment schema, example CSVs, and key-integrity quick check; no trained risk model. |
| [COVID-19 tweet BERT draft](<1_OTHER PROJECTS/Multiclass_classification_model_to_predict_the_sentiment_of_Covid_19_tweets.ipynb>) | BERT fine-tuning and prediction code is present, with **no saved execution or evaluation output**. |
| [Portfolio optimisation PDF](<1_OTHER PROJECTS/104_Portfolio Optimization_MPT.pdf>) | A standalone document; no NIFTY data, price-fetching script, optimization notebook, or reproducible result is present here. |

## Repository guide

The [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md) explains each workflow, formulas, validation boundaries, and practical next steps. Notebooks are independent except the two-stage credit-risk capstone and the CartPole experiment family. Most input datasets are **not committed** and several notebooks use author-specific Windows or Colab paths. There is no root lockfile or unified runner. Open a notebook to inspect saved results; to reproduce one, obtain its cited data, replace local paths, create a compatible environment, and run it deliberately in a fresh kernel. The ALM example has its own [setup guide](ALM_Project/README.md).

## Tools demonstrated

Python, pandas, NumPy, scikit-learn, statsmodels, TensorFlow/Keras, PyTorch, XGBoost, imbalanced-learn, BeautifulSoup, Matplotlib, Seaborn, and Jupyter. Tools vary by project; this is not a single installable package.
