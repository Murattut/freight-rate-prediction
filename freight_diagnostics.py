from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from freight_features import TARGET, canonicalize_categories
from freight_models import SEED, fit_model, predict_frame

def latest_two_month_split(
    frame: pd.DataFrame, *, min_train_months: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split at the latest cutoff with both following calendar months observed."""
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    months = set(dated["date"].dt.to_period("M").dropna().unique())
    eligible = [
        month
        for month in sorted(months)
        if len([observed for observed in months if observed <= month]) >= min_train_months
        and month + 1 in months
        and month + 2 in months
    ]
    if not eligible:
        raise ValueError("no cutoff has both following calendar months and enough training history")
    cutoff = eligible[-1]
    periods = dated["date"].dt.to_period("M")
    train = dated.loc[periods <= cutoff].copy()
    future = dated.loc[periods.isin([cutoff + 1, cutoff + 2])].copy()
    return train, future, str(cutoff)

def make_cold_start_split(
    train_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    seed: int = SEED,
    heldout_category_count: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Hold out deterministic training-only pickup and delivery categories."""
    if heldout_category_count < 1:
        raise ValueError("heldout_category_count must be positive")
    train_dates = pd.to_datetime(train_frame["date"], errors="coerce")
    future_dates = pd.to_datetime(future_frame["date"], errors="coerce")
    if train_dates.isna().any() or future_dates.isna().any():
        raise ValueError("cold-start split requires valid dates")
    if train_dates.max() >= future_dates.min():
        raise ValueError("cold-start split requires future dates after the frozen cutoff")
    normalized_train = canonicalize_categories(train_frame)
    normalized_future = canonicalize_categories(future_frame)
    rng = np.random.default_rng(seed)
    heldout: dict[str, list[str]] = {}
    for column in ("pickup", "delivery"):
        categories = sorted(normalized_train[column].dropna().unique().tolist())
        if len(categories) < heldout_category_count:
            raise ValueError(f"not enough training {column} categories for cold-start split")
        heldout[column] = sorted(
            rng.choice(categories, size=heldout_category_count, replace=False).tolist()
        )
    heldout_union = sorted(set(heldout["pickup"]) | set(heldout["delivery"]))
    removed_train = normalized_train["pickup"].isin(heldout_union) | normalized_train[
        "delivery"
    ].isin(heldout_union)
    stress_mask = normalized_future["pickup"].isin(heldout_union) | normalized_future[
        "delivery"
    ].isin(heldout_union)
    reduced_train = train_frame.loc[~removed_train].copy()
    stress_rows = future_frame.loc[stress_mask].copy()
    if reduced_train.empty or stress_rows.empty:
        raise ValueError("cold-start split produced an empty training or stress subset")
    heldout["all_categories"] = heldout_union
    heldout["n_train_before"] = int(len(train_frame))
    heldout["n_train_after"] = int(len(reduced_train))
    heldout["n_train_removed"] = int(removed_train.sum())
    heldout["n_stress_rows"] = int(len(stress_rows))
    return reduced_train, stress_rows, heldout

def run_cold_start_diagnostics(
    train_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    *,
    selected_model: str,
    baseline_model: str = "catboost_no_market_quote",
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    reduced_train, stress_rows, heldout = make_cold_start_split(
        train_frame, future_frame, seed=seed, heldout_category_count=4
    )
    rows: list[dict[str, object]] = []
    model_names = list(dict.fromkeys([selected_model, baseline_model]))
    train_end = pd.to_datetime(train_frame["date"]).max().to_period("M")
    for model_name in model_names:
        model = fit_model(reduced_train, seed=seed, model_name=model_name)
        for valid_month in sorted(pd.to_datetime(stress_rows["date"]).dt.to_period("M").unique()):
            valid = stress_rows[pd.to_datetime(stress_rows["date"]).dt.to_period("M") == valid_month]
            actual = valid[TARGET].astype(float).to_numpy()
            predictions = predict_frame(model, valid)
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(valid_month.ordinal - train_end.ordinal),
                    "train_end": str(train_end),
                    "valid_month": str(valid_month),
                    "n_train": len(reduced_train),
                    "n_valid": len(valid),
                    "mae": mean_absolute_error(actual, predictions),
                    "rmse": np.sqrt(mean_squared_error(actual, predictions)),
                    "evaluation_role": "development_stress_test",
                }
            )
    return pd.DataFrame(rows), heldout

def build_error_case_audit(
    train_frame: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    selected_model: str,
    cases_per_tail: int = 50,
) -> pd.DataFrame:
    selected = oof[oof["model"] == selected_model].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "case_type", "load_id", "date", "pickup", "delivery", "pickup_lat",
                "pickup_lon", "delivery_lat", "delivery_lon", "distance", "equipment",
                "weight", "actual", "prediction", "residual",
                "absolute_error", "actual_to_predicted_ratio",
            ]
        )
    source_columns = [
        column for column in (
            "load_id", "date", "pickup", "delivery", "pickup_lat", "pickup_lon",
            "delivery_lat", "delivery_lon", "distance", "equipment", "weight"
        ) if column in train_frame
    ]
    audit = selected.merge(train_frame[source_columns], on="load_id", how="left", validate="one_to_one", suffixes=("", "_source"))
    if "date_source" in audit:
        audit["date"] = audit["date_source"]
        audit = audit.drop(columns="date_source")
    for column in (
        "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
        "delivery_lon", "distance", "equipment", "weight"
    ):
        source_column = f"{column}_source"
        if source_column in audit:
            audit[column] = audit[source_column]
            audit = audit.drop(columns=source_column)
    audit["absolute_error"] = (audit["actual"] - audit["prediction"]).abs()
    audit["actual_to_predicted_ratio"] = audit["actual"] / audit["prediction"]
    positive = audit[audit["residual"] > 0].nlargest(cases_per_tail, "residual").copy()
    positive["case_type"] = "positive_residual"
    negative = audit[audit["residual"] < 0].nsmallest(cases_per_tail, "residual").copy()
    negative["case_type"] = "negative_residual"
    result = pd.concat([positive, negative], ignore_index=True)
    return result.drop_duplicates("load_id")
