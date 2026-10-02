# Freight rate prediction

This repository contains my solution to Spotter’s freight-rate assessment. It trains on 48,000 labeled loads from January–October 2025 and writes predictions for the 12,000 November–December loads. The final candidate is a CatBoost model trained to predict rate per mile, with distance-based sample weights. The assessment report and Loom walkthrough are submitted separately; this repository contains the code and prediction files.

## Run it

The project uses Python 3.12 or later and [uv](https://docs.astral.sh/uv/). Place the four CSV files supplied with the assessment in the project root. They are kept out of this public repository. Then run:

```bash
uv sync
uv run python main.py
```

The default run compares three models: Ridge, raw-rate CatBoost, and distance-weighted CatBoost rate per mile. It performs the rolling backtest, fits the selected model on all labeled rows, and writes both prediction files and the diagnostics described below. To run the wider experimental catalog as well, use `uv run python main.py --extended-benchmarks`.

Check the submission files and create the required December chart with the supplied scorer:

```bash
uv run python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december_predictions.csv
```

The scorer writes the December chart to `scorer_results/candidate_december.png`. The assessment report is a separate submission file.

The input paths can be changed with `--train`, `--validation`, `--template`, and `--december-inputs`. The prediction and diagnostic output paths also have command-line options; run `uv run python main.py --help` to see them.

## Data and validation

The labeled file has 48,000 rows dated January 1 through October 31, 2025. The assessment validation file has 12,000 rows dated November 1 through December 31. I used forward, month-ahead folds so each backtest prediction comes from a model trained on earlier dates. May–August are the four folds used to rank eligible models by mean MAE; RMSE is the tie-breaker. September and October are reported as later development checks.

Those later months had already been used in earlier experiments. They are not an untouched holdout, and the earlier results informed which candidates I kept eligible. The current three eligible candidates are ranked using the May–August folds. The CSV keeps the legacy `fold_role=holdout` value for September and October, but in this project it means “later development check.” The November–December labels are hidden, so the assessment score cannot be calculated here.

Validation has more route novelty than the rolling out-of-fold predictions: 736 validation routes do not occur in training, covering 1,461 loads (about 12.2% of the validation rows). Unseen routes make up about 0.71% of the out-of-fold rows. The model uses pickup and delivery, coordinates, equipment, distance, weight, and calendar features, so it can make predictions without an exact route match.

## Model choice

The selected model is `catboost_weighted_rpm_no_market_quote`. It learns `posted_rate / distance` using CatBoost’s MAE objective, then multiplies its prediction by the load’s distance. Each training row receives a weight of `distance / mean(training distance)`. This makes the weighted absolute error on rate per mile proportional to absolute error in the total dollar rate. The model uses the same fold-fitted features as raw-rate CatBoost; it keeps every training row and target.

The CatBoost settings are 450 iterations, depth 8, learning rate 0.06, and L2 leaf regularization 5. The loss is MAE, the random seed is 42, and the fit uses four threads.

The selected feature set includes normalized pickup, delivery, and equipment categories; coordinates and straight-line distance; load distance and distance bands; weight and weight-quality flags; and calendar features. I left `market_index` and `quote_signal` out of final selection because the supplied files do not say when those values became available. That timing matters if the model is meant to predict a rate before a quote is made.

| Candidate | May–August MAE ($) | May–August RMSE ($) | September–October MAE ($) | September–October RMSE ($) |
| --- | ---: | ---: | ---: | ---: |
| Distance-weighted CatBoost rate per mile | 117.17 | 627.48 | 105.93 | 631.48 |
| Raw-rate CatBoost | 122.95 | 628.29 | 118.32 | 634.87 |
| Ridge | 177.58 | 638.00 | 160.64 | 644.48 |

The weighted model’s May–August mean MAE was $117.17, compared with $122.95 for raw-rate CatBoost. At the frozen August cutoff, the one-month September MAE was $105.40 and the two-month October MAE was $115.15; raw-rate CatBoost scored $119.31 and $144.47 at those horizons. These checks were available during development and should be read as comparisons between candidates, not as a fresh estimate of future accuracy.

I also tested a Huber log-rate-per-mile model. It looked better on some early folds, then did worse on the reused later months and the city stress test, so it remains in the extended catalog but is excluded from selection. The decision log records the earlier experiments and why I stopped pursuing route averages, blends, and a separate long-haul model.

## Data quality and error checks

The training file has 300 missing weights and 592 zero or negative weights, including 292 negative values. I retained those loads, marked missing and invalid weights, and let each model handle missing values within its training fold. The file also has 374 missing `market_index` values. No source timestamps are provided for `market_index` or `quote_signal`, which is why neither is used by the selected model.

The diagnostics look at errors from several angles:

- `forecast_horizon_metrics.csv` records one- and two-month forecasts from the same frozen training cutoff where labeled months are available. The August cutoff provides the September and October comparison above.
- `cold_start_metrics.csv` measures performance on later loads involving selected pickup or delivery cities removed from training. For this comparison, both models use the same reduced training set. The selected model’s September/October MAE was $144.73/$134.45; raw-rate CatBoost scored $169.97/$193.94. This is a deliberately concentrated stress test, not an estimate of how often such loads occur.
- `selected_error_diagnostics.json` summarizes selected-model out-of-fold errors. Median absolute error was $37.40, the 99th percentile was $1,371.86, and the worst 1% of rows contributed 96.9% of squared error. A small number of difficult loads therefore dominate RMSE.
- `error_case_audit.csv` lists up to 50 expensive underpredictions and 50 cheap overpredictions with their source fields. High actual-to-predicted ratios are worth inspecting, but they do not establish that a label is wrong. All target rows remain in training.

I also ran one exploratory check to see whether the available features could identify extreme errors. I flagged rows where the earlier raw-rate CatBoost out-of-fold actual-to-predicted ratio was outside 0.5–2, trained a classifier on May–June, and checked July–August. Average precision was 0.014, close to the 0.0138 event rate, and ROC AUC was 0.509. That is weak separation in this one check; it does not explain the errors or show that they are irreducible.

The cold-start and later-month results are retrospective: their labels were already available during development, and the city subset is not prevalence-matched. The audit is a way to find cases for review, not a claim that the remaining errors have been explained.

## Files

The main pipeline writes:

- `validation_predictions.csv` — exactly `load_id,predicted_rate`, one prediction for each validation ID, in template order.
- `december_predictions.csv` — the seven required December columns, including `predicted_rate`.
- `backtest_metrics.csv`, `backtest_oof_predictions.csv`, and `backtest_segment_metrics.csv` — fold scores, out-of-fold predictions, and error summaries by segment.
- `forecast_horizon_metrics.csv`, `cold_start_metrics.csv`, and `cold_start_cities.json` — the frozen-horizon and city stress diagnostics.
- `selected_error_diagnostics.json` and `error_case_audit.csv` — the selected model’s error tail and signed residual examples.
- `run_manifest.json` — the data ranges, model selection record, feature policy, and missing-value counts.

The scorer creates `scorer_results/candidate_december.png` when you run it.

The code is split by responsibility: `main.py` coordinates data checks, backtesting, model selection, and output writing; `freight_features.py` builds and validates features; `freight_models.py` defines and fits candidates; and `freight_diagnostics.py` calculates the validation diagnostics.
