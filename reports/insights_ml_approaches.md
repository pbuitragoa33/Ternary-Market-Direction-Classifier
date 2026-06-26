# Insights ML Approaches (1-day vs 5-day prediction)

The data and whole process was analyzed from the idea of getting an edge for the next day prediction. The analysis was done through a walk-forward validation approach, where the models were trained on a rolling window of 3 years of data and tested on the next 6 months of data, giving a total og 16 folds.

2 experiments were conducted:

* **Experiment 1:** Next day (horizon = 1) BTC price-direction classification with hyperparameter tuning and walk-forward validation across 9 model families. The target depends if the return was above/below/equal to 1%.

* **Experiment 2:** 5 day (horizon = 5) BTC price-direction classification with hyperparameter tuning and walk-forward validation across 9 model families.The target depends if the return was above/below/equal to 2%.


## 1. Problem Definition

The target labels is built from `Close[t + 1]` or `Close[t + 5]`, the raw `Future_Return` column is an "oracle", because it leaks the future, but is used as a ceiling reference not as a feature. The model muest predict the `Target[t]` from infromation available up to and including day `t`. This is, by design, one of the hardest prediction problems in quantitative finance: short-horizon directional forecasting of a highly efficient, heavily arbitraged asset.

## 2. The Class Balance Challenge

* **Rows:** aproximately 3.900 daily observations
* **Span:** from 2015 to 2026 → it covers multiple regimes, but dominated by a structurally bullish trend, anomalous post-2020 macro period. 

This coincides with the central bank liquidity injections (quantitative easing) along with the post-pandemix recovery, which has been a tailwind for BTC and other risk assets. 


* With H = 1, the class balance is:

| Class   | Label | Share |
| ------- | ----- | ----- |
| Neutral |   1   | 39.5% |
| Bullish |   2   | 33.0% |
| Bearish |   0   | 27.5% |

![Class Balance H = 1](../outputs/EDA_content/Target_by_target_distribution.png)


* With H = 5, the class balance is:

| Class   | Label | Share |
| ------- | ----- | ----- |
| Neutral |   1   | 30.6% |
| Bullish |   2   | 39.4% |
| Bearish |   0   | 30.0% |

![Class Balance H = 5](../outputs/EDA_content/Target_by_target_distribution_for_5d_pred.png)


* Reference Baselines

| Baseline                        | Accuracy | F1-Macro |
| ------------------------------- | -------- | -------- |
| Always-Neutral (Majority Class) |    0.395 |    0.190 |
| Random Uniform (1/3 each class) |    0.333 |    0.330 |


The always-majority baseline gives a deceptively low F1-macro (~0.19) because it scores zero on two of three classes. The honest bar for "the model learned something" is F1-macro ≈ 0.33 (random uniform), not 0.19.


## 3. Methodology

* **Primary metric:** F1-macro. Chosen over accuracy (which rewards predicting the majority class) and F1-weighted (which hides minority-class failure). F1-macro weights all three classes equally, so a model that abandons Bearish is penalized.

* **Class balancing:** Custom sample/class weights `{Bearish: 2, Neutral: 1, Bullish: 1}` to push models to attempt the minority Bearish class.

* **Temporal validaty**: Never shuffle the data. 

    - Tuning: `TimeSeriesSplit` (5 expanding folds) inside each Optuna objective. The objective is the mean F1-macro across the 5 folds.
    - Validation: Walk-forward with rolling windows.

* **Why two stages**: Tuning finds good hyperparameters efficiently; walk-forward then stress-tests those hyperparameters across many out-of-sample windows to produce a distribution of performance rather than a single lucky number.

* **Tooling**: Optuna (TPE sampler) for search; Weights & Biases for tracking (one run per model, per-trial metrics, trials tables, and versioned artifacts of the study + best-params JSON). Data flows through an AWS S3 bronze/silver/gold layout; the gold layer holds the final scaled, encoded features.

## 4. Hyperparameter Tuning Results

* **Setup:** 9 model families with up to 500 trials per model and 1 hour timeout, also `TimeSeriesSplit(5)` and custom class weights.

Ranked by the best CV F1-macro:

With **H = 1**, the top 9 models are:

| Rank | Model             | F1-Macro | F1 Std | Accuracy | Trials Completed | Time (s)  |
| ---- | ----------------- | -------- | ------ | -------- | ---------------- | --------- |
|    1 | KNN               |   0.3600 | 0.0467 |   0.389  |              500 |      59   |
|    2 | AdaBoost          |   0.3567 | 0.0397 |   0.403  |              242 |      3607 |
|    3 | Decision Tree     |   0.3523 | 0.0397 |   0.375  |              500 |      123  |
|    4 | LightGBM          |   0.3468 | 0.0231 |   0.371  |              500 |      2685 |
|    5 | Stacking          |   0.3285 | 0.0270 |   0.352  |              42  |      3601 |
|    6 | Random Forest     |   0.3209 | 0.0427 |   0.368  |              500 |      3355 |
|    7 | XGBoost           |   0.3179 | 0.0429 |   0.343  |              116 |      3818 |
|    8 | Gradient Boosting |   0.3107 | 0.0232 |   0.331  |              20  |      3861 | 
|    9 | Hard Voting       |   0.3030 | 0.0504 |   0.350  |              500 |      571  |


With **H = 5**, the top 9 models are:

| Rank | Model             | F1-Macro | F1 Std | Accuracy | Trials Completed | Time (s)   |
| ---- | ----------------- | -------- | ------ | -------- | ---------------- | ---------- |
|    1 | Stacking          |   0.3308 | 0.0526 |   0.361  |              230 |       2258 |
|    2 | KNN               |   0.3073 | 0.0535 |   0.357  |              242 |       3607 |
|    3 | Gradient Boosting |   0.2887 | 0.0598 |   0.335  |              361 |       1892 |
|    4 | Decision Tree     |   0.2767 | 0.0671 |   0.338  |              500 |       1685 |
|    5 | XGBoost           |   0.2728 | 0.0544 |   0.341  |              426 |       3580 |
|    6 | LightGBM          |   0.2727 | 0.0608 |   0.364  |              500 |       3241 |
|    7 | Hard Voting       |   0.2506 | 0.0936 |   0.343  |              500 |       674  |
|    8 | AdaBoost          |   0.2341 | 0.0528 |   0.379  |              222 |       3861 | 
|    9 | Random Forest     |   0.2318 | 0.0439 |   0.351  |              500 |       2458 |


## 5. Walk-Forward Validation Results

* **Train Window**: 3 years
* **Test Window**: next 6 months
* **Slide**: 6 months
* **Folds**: 16
* **Per fold**: F1-macro, accuracy, per-class precision/recall/F1, train-vs-test overfitting gap, and (where `predict_proba` exists) macro one-vs-rest ROC-AUC, PR-AUC, multiclass log-loss and Brier.

With **H = 1**, the results are: 

| Model             |      F1-Macro     | ROC-AUC (OvR) |   PR-AUC  | Overfit Gap | Bearish Recall | Beats Baseline | Log-Loss |
| ----------------- | ----------------- | ------------- | --------- | ----------- | -------------- | -------------- | -------- |
| Gradient Boosting |   0.330 ± 0.043   |     0.539     |   0.382   |       0.639 |          0.319 |      100%      |     1.18 |
| KNN               |   0.322 ± 0.047   |     0.532     |   0.367   |       0.162 |          0.225 |      100%      |     1.20 |
| Stacking          |   0.319 ± 0.039   |     0.499     |   0.336   |      −0.070 |          0.221 |      100%      |    23.50 |
| XGBoost           |   0.313 ± 0.060   |     0.534     |   0.380   |       0.603 |          0.272 |      100%      |     1.23 |
| Decision Tree     |   0.306 ± 0.075   |     0.521     |   0.355   |       0.267 |          0.359 |       94%      |    10.50 |
| LightGBM          |   0.302 ± 0.049   |     0.537     |   0.385   |       0.512 |          0.274 |      100%      |     1.12 |
| Hard Voting       |   0.282 ± 0.055   |      n/a      |    n/a    |       0.242 |          0.219 |       88%      |      n/a |
| AdaBoost          |   0.275 ± 0.064   |     0.548     |   0.391   |       0.120 |          0.135 |       88%      |     1.09 |
| Random Forest     |   0.274 ± 0.079   |     0.559     |   0.402   |       0.261 |          0.192 |       88%      |     1.09 |


- The cleanest verdict is ROC-AUC and it say "no signal". One-vs-rest macro ROC-AUC is threshold- and balance-independent — it measures whether the model can rank days by directional probability. Every model lands between 0.50 and 0.56. The best (Random Forest, 0.559) is barely above coin-flip; Stacking is at 0.4985 — below chance.

- "Beats baselone 100%" is a trap, not a triumph. The baseline is always-majority, whose F1-macro is 0.19 by constructon. Any model that simply spreads predictions across all 3 classes will beat that baseline, not by being right, but by not predicting the others at 0. Against the correct bar (random uniform ≈ 0.33), no model clears it; the leader (Gradient Boosting, 0.330) merely ties chance.

- The F1 ranking is rewarding overfitting, not the skill: 

    - Gradient Boosting "wins" at F1 0.330 but with a gap of 0.639 — it scores ~0.97 on train and 0.33 on test. It memorized the training noise; its test F1 is high by variance, not by learning.
    - XGBoost (gap 0.603) and LightGBM (gap 0.512): same story — heavy memorization.
    - Stacking has a near-zero/negative gap (−0.070), i.e. it generalizes — and its AUC of 0.4985 confirms there was simply nothing to learn.
    - KNN (gap 0.162) and AdaBoost (gap 0.120) are the healthiest generalizers, but their AUC (~0.53–0.55) is still essentially chance.

- The pattern is unmistakable: huge train-test gap + test F1 near chance + AUC near 0.5 → the models memorizes the training noise and are not able to generalize.


With **H = 5**, the results are (respecting the same order of models as above):

| Model             |      F1-Macro     | ROC-AUC (OvR) |   PR-AUC  | Overfit Gap | Bearish Recall | Beats Baseline | Log-Loss |
| ----------------- | ----------------- | ------------- | --------- | ----------- | -------------- | -------------- | -------- |
| Gradient Boosting |   0.288 ± 0.589   |     0.488     |   0.347   |       0.639 |          0.389 |      94%       |     1.18 |
| KNN               |   0.307 ± 0.053   |     0.532     |   0.367   |       0.268 |          0.315 |      100%      |     1.68 |
| Stacking          |   0.330 ± 0.0526  |     0.509     |   0.341   |      -0.113 |          0.328 |      100%      |    23.08 |
| XGBoost           |   0.272 ± 0.067   |     0.517     |   0.372   |       0.685 |          0.409 |      88%       |     1.34 |
| Decision Tree     |   0.276 ± 0.067   |     0.507     |   0.351   |       0.756 |          0.430 |      94%       |    10.78 |
| LightGBM          |   0.272 ± 0.061   |     0.517     |   0.375   |       0.595 |          0.344 |      94%       |     1.16 |
| Hard Voting       |   0.250 ± 0.093   |      n/a      |    n/a    |       0.369 |          0.348 |      88%       |      n/a |
| AdaBoost          |   0.234 ± 0.052   |     0.517     |   0.372   |       0.249 |          0.154 |      88%       |     1.09 |
| Random Forest     |   0.232 ± 0.043   |     0.534     |   0.396   |       0.400 |          0.312 |       88%      |     1.13 |

- The results are similar to H = 1, sometomes even worse. All models are sticked in the 0.50 level of ROC-AUC, notably some model are below the random chance. So, there is no directional information 5 days ahead.

- The models with the highest recall_bullish (knn, adaboost and random forest) are the ones with the highest overfitting gap, so they are not generalizing well. The best generalizer is Stacking, but it is still below random chance.


## 6. Conclusion

Across 9 model families spanning distinct algorithmic paradigms and 2 prediction horizon, all the results converge to the same change-level ceiling: 

    - ROC-AUC ≈ 0.50 ± 0.05 (near randomness)
    - F1-macro ≈ 0.33 ± 0.05 (at or belowrandom uniform)
    - Large overfitting gaps wherever the model is flexible enough to memorize.

This wall belongs to teh data, not the models. There is no exploitable next-day nor 5-day directional signal in BTC with these features, there is no edge or predictive information.

The fact that the horizon was changed from 1 to 5 days was correct, but the answer is unequivocal: the problem was not the horizon. It is that BTC direction, with these features, is not predictable — neither at 1 nor at 5 days.

It's not worth trying deep learning or transformer architectires, because the problem is not the model, it is the data. The features are not informative enough to predict BTC direction at these horizons.

![Walkforward F1-macro Boxplot FOR 1-day](../outputs/walkforward/walkforward_f1_boxplot.png)
![Walkforward F1-macro Boxplot FOR 5-day](../outputs/walkforward/walkforward_5d_f1_boxplot.png)