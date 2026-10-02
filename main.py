from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from freight_features import (
    CATEGORY_COLUMNS, DECEMBER_COLUMNS, DISTANCE_BINS, DISTANCE_LABELS, HOLIDAY_DATES,
    OPTIONAL_FEATURE_COLUMNS, REQUIRED_COMMON_COLUMNS, REQUIRED_ID_COLUMNS,
    REQUIRED_NUMERIC_COLUMNS, TARGET, canonicalize_categories, enrich_chart_inputs,
    make_features, read_data, validate_data_contract,
)
from freight_models import (
    BASE_MODEL_NAMES, DEFAULT_MODEL_NAMES, EXTENDED_MODEL_NAMES, MODEL_NAMES,
    REJECTED_MODEL_NAMES, ROUTE_GROUP_COLUMNS, SEED, BlendedModel, FreightModel, RouteStatistics,
    add_route_statistics, build_model, feature_columns, fit_model,
    fit_route_statistics, predict_frame,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from freight_diagnostics import (
    build_error_case_audit, latest_two_month_split, make_cold_start_split,
    run_cold_start_diagnostics,
)


def segment_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """I compare out-of-fold errors across months and load types."""
    segment_columns = [
        "valid_month",
        "equipment",
        "distance_bucket",
        "route_seen",
        "weight_status",
    ]
    rows: list[dict[str, object]] = []
    for segment in segment_columns:
        if segment not in oof.columns:
            continue
        grouped = oof.groupby(["model", segment], dropna=False)
        for (model, value), group in grouped:
            actual = group["actual"].astype(float)
            prediction = group["prediction"].astype(float)
            rows.append(
                {
                    "model": model,
                    "segment": segment,
                    "segment_value": str(value),
                    "n": len(group),
                    "mae": mean_absolute_error(actual, prediction),
                    "rmse": np.sqrt(mean_squared_error(actual, prediction)),
                    "mean_error": float((actual - prediction).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_run_manifest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    selected_model: str,
    seed: int,
    market_quote_availability_verified: bool = False,
) -> dict[str, object]:
    """I record the data, model choice, and validation setup for this run."""
    def dataset_summary(frame: pd.DataFrame, has_target: bool) -> dict[str, object]:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        summary: dict[str, object] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "date_min": str(dates.min().date()),
            "date_max": str(dates.max().date()),
            "has_target": has_target,
            "duplicate_load_ids": int(frame["load_id"].duplicated().sum()),
            "missing_values": {
                column: int(value)
                for column, value in frame.isna().sum().items()
                if value
            },
        }
        return summary

    if "fold_role" in metrics:
        tuning_metrics = metrics[metrics["fold_role"] == "tune"]
        holdout_metrics = metrics[metrics["fold_role"] == "holdout"]
    else:
        # I keep support for older metric files that do not have fold roles.
        tuning_metrics = metrics
        holdout_metrics = metrics.iloc[0:0]
    summary = tuning_metrics.groupby("model")[["mae", "rmse"]].mean().sort_values(
        ["mae", "rmse"]
    )
    if selected_model not in summary.index:
        raise ValueError(f"selected model {selected_model!r} has no tuning metrics")
    selected_holdout = holdout_metrics[holdout_metrics["model"] == selected_model]
    tuning_months = (
        sorted(tuning_metrics["valid_month"].dropna().astype(str).unique().tolist())
        if "valid_month" in tuning_metrics
        else []
    )
    holdout_months = (
        sorted(holdout_metrics["valid_month"].dropna().astype(str).unique().tolist())
        if "valid_month" in holdout_metrics
        else []
    )
    all_candidates = summary.index.tolist()
    selection_candidates = (
        [name for name in all_candidates if name not in REJECTED_MODEL_NAMES]
        if market_quote_availability_verified
        else [
            name
            for name in all_candidates
            if name.endswith("_no_market_quote") and name not in REJECTED_MODEL_NAMES
        ]
    )
    if selected_model not in selection_candidates:
        raise ValueError(
            f"selected model {selected_model!r} is not eligible under the recorded "
            "feature-availability policy"
        )
    profile = (
        "availability_safe_without_market_quote"
        if selected_model.endswith("_no_market_quote")
        else "market_quote_features_enabled_after_availability_verification"
    )
    return {
        "seed": int(seed),
        "selected_model": selected_model,
        "candidate_models": all_candidates,
        "selection_candidate_models": selection_candidates,
        "benchmark_only_models": [
            name for name in all_candidates if name not in selection_candidates
        ],
        "rejected_models": list(REJECTED_MODEL_NAMES),
        "rejected_model_reasons": {
            "huber_rpm_no_market_quote": (
                "early tuning gains did not persist on reused later-development and cold-start diagnostics"
            )
        },
        "eligibility_review_note": (
            "Weighted CatBoost inclusion and Huber rejection reflect review of reused later-development and cold-start diagnostics; "
            "ranking among eligible models in this run uses tuning folds only."
        ),
        "selection_metric": "mean_tuning_mae_then_rmse_with_availability_safe_feature_gate",
        "evaluation_protocol": {
            "tuning_folds": tuning_months,
            "holdout_folds": holdout_months,
            "selection_uses_holdout_metrics": False,
        },
        "metrics": {
            "best_mae": float(summary.loc[selected_model, "mae"]),
            "best_rmse": float(summary.loc[selected_model, "rmse"]),
            "tuning_mae": float(summary.loc[selected_model, "mae"]),
            "tuning_rmse": float(summary.loc[selected_model, "rmse"]),
            "holdout_mae": (
                float(selected_holdout["mae"].mean()) if not selected_holdout.empty else None
            ),
            "holdout_rmse": (
                float(selected_holdout["rmse"].mean()) if not selected_holdout.empty else None
            ),
            "fold_rows": int(len(metrics)),
        },
        "train": dataset_summary(train, has_target=True),
        "validation": dataset_summary(validation, has_target=False),
        "feature_policy": {
            "canonical_categories": list(CATEGORY_COLUMNS),
            "invalid_weight_policy": "retain_with_indicators_and_fold_imputation",
            "route_statistics": "leave_one_out_training_and_smoothed_recency_weighted_inference",
            "selected_feature_profile": profile,
            "selected_model_feature_profile": (
                "huber_plain_log_rate_per_mile"
                if selected_model.removesuffix("_no_market_quote") == "huber_rpm"
                else "catboost_distance_weighted_rate_per_mile"
                if selected_model.removesuffix("_no_market_quote") == "catboost_weighted_rpm"
                else "general_freight_rate_features"
            ),
            "market_index_and_quote_signal_source_timestamps_verified": bool(
                market_quote_availability_verified
            ),
            "feature_availability_assumption": (
                "operator asserted point-in-time availability for market_index and quote_signal"
                if market_quote_availability_verified
                else "assessment files do not establish point-in-time availability; these features are excluded from final selection"
            ),
        },
    }


def write_run_manifest(manifest: dict[str, object], output_path: str | Path) -> None:
    """I save the run details as a readable JSON file."""
    Path(output_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def temporal_splits(
    frame: pd.DataFrame, min_train_months: int = 1
) -> list[tuple[pd.DataFrame, pd.DataFrame, str, str]]:
    """I make month-ahead folds so training never includes future rows."""
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    months = sorted(dated["date"].dt.to_period("M").dropna().unique())
    folds = []
    for index in range(min_train_months, len(months)):
        train_months = months[:index]
        valid_month = months[index]
        train = dated[dated["date"].dt.to_period("M").isin(train_months)].copy()
        valid = dated[dated["date"].dt.to_period("M") == valid_month].copy()
        folds.append((train, valid, str(train_months[-1]), str(valid_month)))
    return folds










def run_forecast_diagnostics(
    train_frame: pd.DataFrame,
    *,
    model_name: str,
    min_train_months: int = 4,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate one frozen model fit at each cutoff at calendar horizons 1 and 2."""
    dated = train_frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated["_month"] = dated["date"].dt.to_period("M")
    observed_months = set(dated["_month"].dropna().unique())
    rows: list[dict[str, object]] = []
    for train_end in sorted(observed_months):
        train_with_month = dated[dated["_month"] <= train_end]
        if train_with_month["_month"].nunique() < min_train_months:
            continue
        train = train_with_month.drop(columns="_month")
        targets = [
            (horizon, train_end + horizon)
            for horizon in (1, 2)
            if train_end + horizon in observed_months
        ]
        if not targets:
            continue
        model = fit_model(train, seed=seed, model_name=model_name)
        for horizon, valid_month in targets:
            valid = dated[dated["_month"] == valid_month].drop(columns="_month")
            actual = valid[TARGET].astype(float).to_numpy()
            predictions = predict_frame(model, valid)
            rows.append(
                {
                    "model": model_name,
                    "train_end": str(train_end),
                    "valid_month": str(valid_month),
                    "horizon_months": horizon,
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "mae": mean_absolute_error(actual, predictions),
                    "rmse": np.sqrt(mean_squared_error(actual, predictions)),
                    "evaluation_role": "development_diagnostic",
                }
            )
    return pd.DataFrame(rows)


def summarize_selected_errors(
    oof: pd.DataFrame, *, selected_model: str
) -> dict[str, object]:
    """Summarize pooled absolute errors and the selected model's worst 1% tail."""
    selected = oof[oof["model"] == selected_model] if "model" in oof else oof.iloc[0:0]
    if selected.empty:
        return {
            "model": selected_model,
            "n": 0,
            "mae": None,
            "pooled_rmse": None,
            "median_absolute_error": None,
            "p95_absolute_error": None,
            "p99_absolute_error": None,
            "worst_1pct_n": 0,
            "worst_1pct_squared_error_share": 0.0,
        }
    errors = (
        selected["actual"].astype(float).to_numpy()
        - selected["prediction"].astype(float).to_numpy()
    )
    absolute = np.abs(errors)
    squared = errors**2
    tail_n = max(1, int(np.ceil(0.01 * len(errors))))
    total_squared = float(squared.sum())
    share = (
        float(np.sort(squared)[-tail_n:].sum() / total_squared)
        if total_squared > 0
        else 0.0
    )
    return {
        "model": selected_model,
        "n": int(len(errors)),
        "mae": float(absolute.mean()),
        "pooled_rmse": float(np.sqrt(np.mean(squared))),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "p99_absolute_error": float(np.percentile(absolute, 99)),
        "worst_1pct_n": tail_n,
        "worst_1pct_squared_error_share": share,
    }


def run_backtest(
    train_frame: pd.DataFrame,
    seed: int = SEED,
    *,
    min_train_months: int = 4,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    holdout_fold_count: int = 2,
    market_quote_availability_verified: bool = False,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """I choose a model on early folds and check it on later folds."""
    folds = temporal_splits(train_frame, min_train_months=min_train_months)
    if not folds:
        raise ValueError("at least one validation month is required for the backtest")
    if holdout_fold_count < 0 or holdout_fold_count >= len(folds):
        raise ValueError("holdout_fold_count must leave at least one tuning fold")

    requested = tuple(model_names)
    blend_names = {
        "blend",
        "catboost_hgb_rpm_blend",
        "catboost_hgb_rpm_blend_no_market_quote",
    }
    base_names = tuple(name for name in requested if name not in blend_names)
    rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    for fold_index, (train, valid, train_end, valid_month) in enumerate(folds):
        fold_role = "holdout" if fold_index >= len(folds) - holdout_fold_count else "tune"
        predictions_by_model: dict[str, np.ndarray] = {}
        actual = valid[TARGET].astype(float).to_numpy()
        valid_features = make_features(valid)
        train_routes = set(make_features(train)["route_key"].astype(str))
        route_seen = valid_features["route_key"].astype(str).isin(train_routes)
        weight_status = np.where(
            valid_features["weight_missing"] > 0,
            "missing",
            np.where(valid_features["weight_invalid"] > 0, "invalid", "valid"),
        )
        for model_name in base_names:
            model = fit_model(train, seed=seed, model_name=model_name)
            predictions_by_model[model_name] = predict_frame(model, valid)

        if "blend" in requested:
            if not {"ridge", "hist_gradient_boosting"}.issubset(predictions_by_model):
                raise ValueError("blend requires ridge and hist_gradient_boosting candidates")
            predictions_by_model["blend"] = (
                0.3 * predictions_by_model["ridge"]
                + 0.7 * predictions_by_model["hist_gradient_boosting"]
            )
        if "catboost_hgb_rpm_blend" in requested:
            if not {"catboost", "hist_gradient_boosting_rpm"}.issubset(predictions_by_model):
                raise ValueError(
                    "catboost_hgb_rpm_blend requires catboost and hist_gradient_boosting_rpm"
                )
            predictions_by_model["catboost_hgb_rpm_blend"] = (
                0.30 * predictions_by_model["catboost"]
                + 0.70 * predictions_by_model["hist_gradient_boosting_rpm"]
            )
        safe_blend = "catboost_hgb_rpm_blend_no_market_quote"
        if safe_blend in requested:
            safe_components = {
                "catboost_no_market_quote",
                "hist_gradient_boosting_rpm_no_market_quote",
            }
            if not safe_components.issubset(predictions_by_model):
                raise ValueError(f"{safe_blend} requires {sorted(safe_components)}")
            predictions_by_model[safe_blend] = (
                0.30 * predictions_by_model["catboost_no_market_quote"]
                + 0.70
                * predictions_by_model["hist_gradient_boosting_rpm_no_market_quote"]
            )

        for model_name in requested:
            predictions = predictions_by_model[model_name]
            rows.append(
                {
                    "model": model_name,
                    "train_end": train_end,
                    "valid_month": valid_month,
                    "n_train": len(train),
                    "n_valid": len(valid),
                    "fold_role": fold_role,
                    "mae": mean_absolute_error(actual, predictions),
                    "rmse": np.sqrt(mean_squared_error(actual, predictions)),
                }
            )
            identifiers = (
                valid["load_id"].astype(str)
                if "load_id" in valid
                else valid.index.astype(str)
            )
            for load_id, date, observed, prediction, equipment, distance_bucket, seen, weight_state in zip(
                identifiers,
                valid["date"],
                actual,
                predictions,
                valid_features["equipment"],
                valid_features["distance_bin"],
                route_seen,
                weight_status,
            ):
                oof_rows.append(
                    {
                        "load_id": load_id,
                        "date": date,
                        "model": model_name,
                        "actual": observed,
                        "prediction": prediction,
                        "residual": observed - prediction,
                        "valid_month": valid_month,
                        "fold_role": fold_role,
                        "equipment": equipment,
                        "distance_bucket": distance_bucket,
                        "route_seen": "seen" if seen else "unseen",
                        "weight_status": weight_state,
                    }
                )

    metrics = pd.DataFrame(rows)
    tuning_metrics = metrics[metrics["fold_role"] == "tune"]
    if tuning_metrics.empty:
        raise ValueError("at least one tuning fold is required for model selection")
    if market_quote_availability_verified:
        selection_metrics = tuning_metrics[~tuning_metrics["model"].isin(REJECTED_MODEL_NAMES)]
    else:
        selection_metrics = tuning_metrics[
            tuning_metrics["model"].str.endswith("_no_market_quote")
            & ~tuning_metrics["model"].isin(REJECTED_MODEL_NAMES)
        ]
    if selection_metrics.empty:
        raise ValueError(
            "no availability-safe candidates were evaluated; include a model ending "
            "in '_no_market_quote' or verify market/quote feature availability"
        )
    summary = selection_metrics.groupby("model", as_index=False)[["mae", "rmse"]].mean()
    best_model = summary.sort_values(["mae", "rmse"]).iloc[0]["model"]
    return metrics, str(best_model), pd.DataFrame(oof_rows)


def write_submission(
    template: pd.DataFrame,
    predictions: np.ndarray,
    output_path: str | Path,
    *,
    prediction_ids: Sequence[object],
) -> None:
    """I write predictions in the two-column format required by the assessment."""
    raw_ids = template["load_id"]
    if raw_ids.isna().any() or raw_ids.duplicated().any():
        raise ValueError("template contains missing or duplicate load_id values")
    ids = raw_ids.astype(str).reset_index(drop=True)
    if ids.duplicated().any():
        raise ValueError("template contains missing or duplicate load_id values")
    values = np.asarray(predictions, dtype=float)
    predicted_ids = pd.Series(prediction_ids)
    if predicted_ids.isna().any() or predicted_ids.duplicated().any():
        raise ValueError("predictions contain missing or duplicate load_id values")
    predicted_ids = predicted_ids.astype(str).reset_index(drop=True)
    if predicted_ids.duplicated().any():
        raise ValueError("predictions contain missing or duplicate load_id values")
    if len(predicted_ids) != len(values):
        raise ValueError("prediction ID count does not match prediction count")
    if set(ids) != set(predicted_ids):
        raise ValueError("template and prediction ID sets do not match")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("predictions must be finite and positive")
    by_id = pd.Series(values, index=predicted_ids)
    values = by_id.loc[ids].to_numpy(dtype=float)
    pd.DataFrame({"load_id": ids, "predicted_rate": values}).to_csv(
        output_path, index=False
    )


def write_december_predictions(
    frame: pd.DataFrame, predictions: np.ndarray, output_path: str | Path
) -> None:
    """I add predictions while keeping the December input columns unchanged."""
    if list(frame.columns) != DECEMBER_COLUMNS:
        raise ValueError(f"December input columns must be {DECEMBER_COLUMNS}")
    if len(frame) != len(predictions):
        raise ValueError("December prediction count does not match input row count")
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("December predictions must be finite and positive")
    output = frame.copy()
    output["predicted_rate"] = values
    output.to_csv(output_path, index=False)


def main() -> None:
    """I run the backtest, fit the selected model, and write its outputs."""
    parser = argparse.ArgumentParser(description="Train and run the freight-rate model.")
    parser.add_argument("--train", default="train-test.csv")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--template", default="validation-predictions-template.csv")
    parser.add_argument("--december-inputs", default="december-chart-inputs.csv")
    parser.add_argument("--prediction-output", default="validation_predictions.csv")
    parser.add_argument("--december-output", default="december_predictions.csv")
    parser.add_argument("--metrics-output", default="backtest_metrics.csv")
    parser.add_argument("--oof-output", default="backtest_oof_predictions.csv")
    parser.add_argument("--segment-metrics-output", default="backtest_segment_metrics.csv")
    parser.add_argument("--manifest-output", default="run_manifest.json")
    parser.add_argument("--forecast-metrics-output", default="forecast_horizon_metrics.csv")
    parser.add_argument("--error-diagnostics-output", default="selected_error_diagnostics.json")
    parser.add_argument("--cold-start-metrics-output", default="cold_start_metrics.csv")
    parser.add_argument("--cold-start-cities-output", default="cold_start_cities.json")
    parser.add_argument("--error-case-audit-output", default="error_case_audit.csv")
    parser.add_argument(
        "--extended-benchmarks",
        action="store_true",
        help="run the full original candidate catalog in addition to the compact defaults",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--market-quote-availability-verified",
        action="store_true",
        help=(
            "allow market_index and quote_signal candidates into final selection only "
            "when their point-in-time availability has been verified"
        ),
    )
    args = parser.parse_args()

    train = read_data(args.train)
    validation = read_data(args.validation)
    template = pd.read_csv(args.template)
    december = pd.read_csv(args.december_inputs, parse_dates=["date"])
    december_model_input = enrich_chart_inputs(december, train)

    metrics, best_model, oof = run_backtest(
        train,
        seed=args.seed,
        model_names=EXTENDED_MODEL_NAMES if args.extended_benchmarks else DEFAULT_MODEL_NAMES,
        market_quote_availability_verified=args.market_quote_availability_verified,
    )
    metrics.to_csv(args.metrics_output, index=False)
    oof.to_csv(args.oof_output, index=False)
    segment_metrics(oof).to_csv(args.segment_metrics_output, index=False)
    run_forecast_diagnostics(
        train, model_name=best_model, min_train_months=4, seed=args.seed
    ).to_csv(args.forecast_metrics_output, index=False)
    Path(args.error_diagnostics_output).write_text(
        json.dumps(summarize_selected_errors(oof, selected_model=best_model), indent=2) + "\n"
    )
    cutoff_train, cutoff_future, _ = latest_two_month_split(train, min_train_months=4)
    cold_metrics, cold_cities = run_cold_start_diagnostics(
        cutoff_train,
        cutoff_future,
        selected_model=best_model,
        seed=args.seed,
    )
    cold_metrics.to_csv(args.cold_start_metrics_output, index=False)
    Path(args.cold_start_cities_output).write_text(
        json.dumps(cold_cities, indent=2, sort_keys=True) + "\n"
    )
    build_error_case_audit(train, oof, selected_model=best_model).to_csv(
        args.error_case_audit_output, index=False
    )
    write_run_manifest(
        build_run_manifest(
            train,
            validation,
            metrics,
            selected_model=best_model,
            seed=args.seed,
            market_quote_availability_verified=args.market_quote_availability_verified,
        ),
        args.manifest_output,
    )
    model = fit_model(train, seed=args.seed, model_name=best_model)

    write_submission(
        template,
        predict_frame(model, validation),
        args.prediction_output,
        prediction_ids=validation["load_id"],
    )
    write_december_predictions(
        december,
        predict_frame(model, december_model_input),
        args.december_output,
    )

    summary = (
        metrics.groupby(["fold_role", "model"])[["mae", "rmse"]]
        .mean()
        .sort_values(["mae", "rmse"])
    )
    print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))
    print(f"Selected model: {best_model}")
    print(f"Wrote {Path(args.prediction_output)}")
    print(f"Wrote {Path(args.december_output)}")
    print(f"Wrote {Path(args.oof_output)}")
    print(f"Wrote {Path(args.segment_metrics_output)}")
    print(f"Wrote {Path(args.forecast_metrics_output)}")
    print(f"Wrote {Path(args.cold_start_metrics_output)}")
    print(f"Wrote {Path(args.cold_start_cities_output)}")
    print(f"Wrote {Path(args.error_case_audit_output)}")
    print(f"Wrote {Path(args.manifest_output)}")


if __name__ == "__main__":
    main()
