from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "posted_rate"

REQUIRED_COMMON_COLUMNS = {
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
}

OPTIONAL_FEATURE_COLUMNS = {"market_index", "quote_signal"}

DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]

DISTANCE_BINS = [-np.inf, 100, 300, 500, 800, 1200, 2000, np.inf]

DISTANCE_LABELS = [
    "under_100",
    "100_300",
    "300_500",
    "500_800",
    "800_1200",
    "1200_2000",
    "over_2000",
]

HOLIDAY_DATES = pd.to_datetime(
    [
        "2025-01-01",
        "2025-01-20",
        "2025-05-26",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
        "2025-12-25",
    ]
)

CATEGORY_COLUMNS = ("pickup", "delivery", "equipment")

REQUIRED_ID_COLUMNS = {"load_id"}

REQUIRED_NUMERIC_COLUMNS = (
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
)

def canonicalize_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """I normalize category values and leave the input frame unchanged."""
    result = frame.copy()
    for column in CATEGORY_COLUMNS:
        if column in result:
            result[column] = (
                result[column]
                .astype("string")
                .str.normalize("NFKC")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .str.casefold()
                .fillna("__missing__")
            )
    return result

def validate_data_contract(
    frame: pd.DataFrame,
    *,
    source: str = "data",
    require_target: bool = False,
    require_id: bool = True,
) -> None:
    """I check the input columns and values before making features."""
    missing = REQUIRED_COMMON_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    if require_id and not REQUIRED_ID_COLUMNS.issubset(frame.columns):
        raise ValueError(f"{source} is missing required columns: ['load_id']")
    if require_target and TARGET not in frame.columns:
        raise ValueError(f"{source} is missing required target column {TARGET!r}")
    if require_id and frame["load_id"].isna().any():
        raise ValueError(f"{source} contains missing load_id values")
    if require_id and frame["load_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate load_id values")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{source} contains invalid dates")
    for column in REQUIRED_NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{source} contains invalid numeric values in {column}")
    for column in OPTIONAL_FEATURE_COLUMNS.intersection(frame.columns):
        raw = frame[column]
        values = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & values.isna()
        if invalid.any() or not np.isfinite(values.dropna()).all():
            raise ValueError(f"{source} contains invalid numeric values in {column}")
    if (pd.to_numeric(frame["distance"], errors="coerce") <= 0).any():
        raise ValueError(f"{source} contains non-positive distance values")
    for column in ("pickup_lat", "delivery_lat"):
        if ((frame[column] < -90) | (frame[column] > 90)).any():
            raise ValueError(f"{source} contains out-of-range latitude values in {column}")
    for column in ("pickup_lon", "delivery_lon"):
        if ((frame[column] < -180) | (frame[column] > 180)).any():
            raise ValueError(f"{source} contains out-of-range longitude values in {column}")
    if require_target:
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError(f"{source} contains invalid target values")

def read_data(path: str | Path) -> pd.DataFrame:
    """I read the CSV, parse its dates, and check its columns."""
    frame = pd.read_csv(path, parse_dates=["date"])
    validate_data_contract(
        frame,
        source=str(path),
        require_target=TARGET in frame.columns,
        require_id=True,
    )
    return frame

def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """I turn each load into features for training and prediction."""
    result = canonicalize_categories(frame)
    date = pd.to_datetime(result["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("date contains invalid values")

    distance = pd.to_numeric(result["distance"], errors="coerce")
    positive_distance = distance.where(distance > 0)
    weight = pd.to_numeric(result["weight"], errors="coerce")

    result["days_since_start"] = (date - pd.Timestamp("2025-01-01")).dt.days.astype(float)
    result["month_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    result["month_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    result["dow_sin"] = np.sin(2 * np.pi * date.dt.dayofweek / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * date.dt.dayofweek / 7.0)
    result["is_weekend"] = (date.dt.dayofweek >= 5).astype(float)
    result["is_holiday"] = date.dt.normalize().isin(HOLIDAY_DATES).astype(float)
    result["is_month_end"] = date.dt.is_month_end.astype(float)

    result["log_distance"] = np.log1p(distance.clip(lower=0))
    result["distance_bin"] = (
        pd.cut(distance, bins=DISTANCE_BINS, labels=DISTANCE_LABELS, include_lowest=True)
        .astype("string")
        .fillna("unknown")
        .astype(str)
    )

    result["delta_lat"] = result["delivery_lat"] - result["pickup_lat"]
    result["delta_lon"] = result["delivery_lon"] - result["pickup_lon"]
    result["geo_distance"] = np.sqrt(result["delta_lat"] ** 2 + result["delta_lon"] ** 2)

    pickup_lat = np.radians(pd.to_numeric(result["pickup_lat"], errors="coerce"))
    pickup_lon = np.radians(pd.to_numeric(result["pickup_lon"], errors="coerce"))
    delivery_lat = np.radians(pd.to_numeric(result["delivery_lat"], errors="coerce"))
    delivery_lon = np.radians(pd.to_numeric(result["delivery_lon"], errors="coerce"))
    delta_lat = delivery_lat - pickup_lat
    delta_lon = delivery_lon - pickup_lon
    haversine_term = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(delivery_lat) * np.sin(delta_lon / 2.0) ** 2
    )
    result["haversine_miles"] = 3958.7613 * 2 * np.arcsin(np.sqrt(haversine_term.clip(0, 1)))

    result["weight_missing"] = weight.isna().astype(float)
    result["weight_invalid"] = (weight.isna() | (weight <= 0)).astype(float)
    result["weight_clean"] = weight.where(weight > 0)
    result["weight_per_mile"] = result["weight_clean"] / positive_distance
    for optional_column in OPTIONAL_FEATURE_COLUMNS:
        if optional_column not in result:
            result[optional_column] = np.nan
    result["market_index_missing"] = result["market_index"].isna().astype(float)

    pickup = result["pickup"].astype("string").fillna("__missing__")
    delivery = result["delivery"].astype("string").fillna("__missing__")
    result["route_key"] = (pickup + " -> " + delivery).astype(str)

    return result.drop(columns=["load_id", "date", TARGET, "weight"], errors="ignore")

def enrich_chart_inputs(frame: pd.DataFrame, reference_frame: pd.DataFrame) -> pd.DataFrame:
    """I fill fields that are missing from the scorer's compact December file."""
    result = frame.copy()
    reference = reference_frame.copy()
    result_keys = canonicalize_categories(result)
    reference_keys = canonicalize_categories(reference)

    pickup_lookup = reference.assign(_pickup_key=reference_keys["pickup"]).groupby(
        "_pickup_key"
    )[["pickup_lat", "pickup_lon"]].first()
    delivery_lookup = reference.assign(_delivery_key=reference_keys["delivery"]).groupby(
        "_delivery_key"
    )[["delivery_lat", "delivery_lon"]].first()
    coordinate_sources = {
        "pickup_lat": pickup_lookup["pickup_lat"],
        "pickup_lon": pickup_lookup["pickup_lon"],
        "delivery_lat": delivery_lookup["delivery_lat"],
        "delivery_lon": delivery_lookup["delivery_lon"],
    }
    for column, lookup in coordinate_sources.items():
        key = "pickup" if column.startswith("pickup_") else "delivery"
        key_values = result_keys[key]
        if column not in result:
            result[column] = key_values.map(lookup)
        else:
            result[column] = result[column].fillna(key_values.map(lookup))
        result[column] = result[column].fillna(reference[column].median())

    for column in ["market_index", "quote_signal"]:
        reference_median = (
            pd.to_numeric(reference[column], errors="coerce").median()
            if column in reference
            else np.nan
        )
        if column not in result:
            result[column] = reference_median
        else:
            result[column] = result[column].fillna(reference_median)
    return result
