from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from freight_features import CATEGORY_COLUMNS, TARGET, make_features

SEED = 42

BASE_MODEL_NAMES = (
    "ridge",
    "hist_gradient_boosting",
    "catboost",
    "hist_gradient_boosting_rpm",
    "catboost_rpm",
    "catboost_route_rpm",
    "catboost_no_market_quote",
    "hist_gradient_boosting_rpm_no_market_quote",
    "huber_rpm_no_market_quote",
    "catboost_weighted_rpm_no_market_quote",
)
EXTENDED_MODEL_NAMES = (
    "ridge",
    "hist_gradient_boosting",
    "catboost",
    "hist_gradient_boosting_rpm",
    "catboost_rpm",
    "catboost_route_rpm",
    "catboost_no_market_quote",
    "ridge_no_market_quote",
    "hist_gradient_boosting_rpm_no_market_quote",
    "blend",
    "catboost_hgb_rpm_blend",
    "catboost_hgb_rpm_blend_no_market_quote",
    "huber_rpm_no_market_quote",
    "catboost_weighted_rpm_no_market_quote",
)
MODEL_NAMES = EXTENDED_MODEL_NAMES
DEFAULT_MODEL_NAMES = (
    "ridge_no_market_quote",
    "catboost_no_market_quote",
    "catboost_weighted_rpm_no_market_quote",
)
REJECTED_MODEL_NAMES = ("huber_rpm_no_market_quote",)
HUBER_NUMERIC_FEATURES = (
    "log_distance",
    "weight_clean",
    "weight_missing",
    "weight_invalid",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",
)
HUBER_CATEGORICAL_FEATURES = ("pickup", "delivery", "equipment")

ROUTE_GROUP_COLUMNS = {
    "route": "route_key",
    "pickup": "pickup",
    "delivery": "delivery",
    "equipment": "equipment",
}

def feature_columns(
    *,
    include_route: bool = False,
    include_route_statistics: bool = False,
    include_market_quote: bool = True,
) -> tuple[list[str], list[str]]:
    numeric = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight_clean",
        "weight_per_mile",
        "weight_missing",
        "weight_invalid",
        "days_since_start",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_holiday",
        "is_month_end",
        "log_distance",
        "delta_lat",
        "delta_lon",
        "geo_distance",
        "haversine_miles",
    ]
    if include_market_quote:
        numeric.extend(["market_index", "quote_signal", "market_index_missing"])
    categorical = ["pickup", "delivery", "equipment", "distance_bin"]
    if include_route:
        categorical.append("route_key")
    if include_route_statistics:
        numeric.extend(
            [
                "route_target_mean",
                "pickup_target_mean",
                "delivery_target_mean",
                "equipment_target_mean",
                "route_count",
                "pickup_count",
                "delivery_count",
                "equipment_count",
                "route_age_days",
                "pickup_age_days",
                "delivery_age_days",
                "equipment_age_days",
            ]
        )
    return numeric, categorical

@dataclass(frozen=True)
class RouteStatistics:
    global_mean: float
    mappings: dict[str, dict[str, tuple[float, float, int, pd.Timestamp]]]
    smoothing: float = 0.0
    half_life_days: float | None = None
    reference_date: pd.Timestamp | None = None

def _decay_weights(
    dates: pd.Series, reference_date: pd.Timestamp, half_life_days: float | None
) -> np.ndarray:
    age_days = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0).to_numpy()
    if half_life_days is None:
        return np.ones(len(dates), dtype=float)
    return np.exp(-np.log(2.0) * age_days / half_life_days)

def fit_route_statistics(
    frame: pd.DataFrame,
    target: Sequence[float],
    *,
    smoothing: float = 0.0,
    half_life_days: float | None = None,
) -> RouteStatistics:
    """I calculate route statistics from the training fold only."""
    features = make_features(frame)
    values = np.asarray(target, dtype=float)
    if len(values) != len(features):
        raise ValueError("route-statistic target length does not match frame")
    if not np.isfinite(values).all():
        raise ValueError("route-statistic target must be finite")
    if smoothing < 0 or (half_life_days is not None and half_life_days <= 0):
        raise ValueError("smoothing must be non-negative and half_life_days must be positive")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    reference_date = dates.max()
    weights = _decay_weights(dates, reference_date, half_life_days)
    global_mean = float(np.average(values, weights=weights))

    mappings: dict[str, dict[str, tuple[float, float, int, pd.Timestamp]]] = {}
    for group_name, column in ROUTE_GROUP_COLUMNS.items():
        grouped = pd.DataFrame(
            {
                "key": features[column].astype(str),
                "target_weighted": values * weights,
                "weight": weights,
                "count": 1,
                "date": dates,
            }
        ).groupby("key").agg(
            weighted_sum=("target_weighted", "sum"),
            total_weight=("weight", "sum"),
            count=("count", "sum"),
            last_date=("date", "max"),
        )
        mappings[group_name] = {
            str(key): (
                float(row["weighted_sum"]),
                float(row["total_weight"]),
                int(row["count"]),
                pd.Timestamp(row["last_date"]),
            )
            for key, row in grouped.iterrows()
        }
    return RouteStatistics(
        global_mean=global_mean,
        mappings=mappings,
        smoothing=float(smoothing),
        half_life_days=half_life_days,
        reference_date=pd.Timestamp(reference_date),
    )

def add_route_statistics(
    frame: pd.DataFrame,
    statistics: RouteStatistics,
    *,
    target: Sequence[float] | None = None,
    leave_one_out: bool = False,
) -> pd.DataFrame:
    """I add target statistics without using a training row's own target."""
    features = make_features(frame)
    values = None if target is None else np.asarray(target, dtype=float)
    if leave_one_out and values is None:
        raise ValueError("leave_one_out route statistics require target values")
    if values is not None and len(values) != len(features):
        raise ValueError("route-statistic target length does not match frame")

    for group_name, column in ROUTE_GROUP_COLUMNS.items():
        mapping = statistics.mappings[group_name]
        keys = features[column].astype(str)
        sums = keys.map({key: value[0] for key, value in mapping.items()}).fillna(0.0).to_numpy()
        weights = keys.map({key: value[1] for key, value in mapping.items()}).fillna(0.0).to_numpy()
        counts = keys.map({key: value[2] for key, value in mapping.items()}).fillna(0.0).to_numpy()
        last_dates = pd.to_datetime(
            keys.map({key: value[3] for key, value in mapping.items()}), errors="coerce"
        )
        dates = pd.to_datetime(frame["date"], errors="coerce")
        row_weights = _decay_weights(
            dates,
            statistics.reference_date or dates.max(),
            statistics.half_life_days,
        )
        if leave_one_out:
            denominator = weights - row_weights
            numerator = sums - row_weights * values
            encoded = np.full(len(keys), statistics.global_mean, dtype=float)
            np.divide(
                numerator + statistics.smoothing * statistics.global_mean,
                denominator + statistics.smoothing,
                out=encoded,
                where=(denominator + statistics.smoothing) > 0,
            )
        else:
            encoded = np.full(len(keys), statistics.global_mean, dtype=float)
            np.divide(
                sums + statistics.smoothing * statistics.global_mean,
                weights + statistics.smoothing,
                out=encoded,
                where=(weights + statistics.smoothing) > 0,
            )
        features[f"{group_name}_target_mean"] = encoded
        features[f"{group_name}_count"] = counts
        age_days = (dates - last_dates).dt.days
        features[f"{group_name}_age_days"] = age_days.fillna(9999.0).clip(lower=0)
    return features

def _catboost_features(
    features: pd.DataFrame,
    *,
    include_route: bool,
    include_route_statistics: bool,
    include_market_quote: bool = True,
) -> pd.DataFrame:
    numeric, categorical = feature_columns(
        include_route=include_route,
        include_route_statistics=include_route_statistics,
        include_market_quote=include_market_quote,
    )
    selected = features[numeric + categorical].copy()
    for column in categorical:
        selected[column] = selected[column].astype("string").fillna("__missing__").astype(str)
    return selected

def _catboost_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=450,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=5.0,
        random_seed=seed,
        random_strength=0.5,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )

def build_model(
    model_name: str,
    seed: int = SEED,
    *,
    include_market_quote: bool = True,
) -> Any:
    if model_name.endswith("_no_market_quote"):
        model_name = model_name.removesuffix("_no_market_quote")
        include_market_quote = False
    valid_core_names = {name.removesuffix("_no_market_quote") for name in BASE_MODEL_NAMES}
    if model_name not in valid_core_names:
        raise ValueError(f"unknown base model {model_name!r}; choose from {BASE_MODEL_NAMES}")

    if model_name.startswith("catboost"):
        return _catboost_model(seed)

    if model_name == "huber_rpm":
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_transformer, list(HUBER_NUMERIC_FEATURES)),
                ("categorical", categorical_transformer, list(HUBER_CATEGORICAL_FEATURES)),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=300, tol=1e-5)),
            ]
        )

    numeric, categorical = feature_columns(include_market_quote=include_market_quote)
    if model_name == "ridge":
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        estimator = Ridge(alpha=10.0)
    else:
        numeric_transformer = SimpleImputer(strategy="median")
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=2.0,
            random_state=seed,
        )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric),
            ("categorical", categorical_transformer, categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

@dataclass
class FreightModel:
    estimator: Any
    model_name: str
    route_statistics: RouteStatistics | None = None
    target_mode: str = "raw"
    include_market_quote: bool = True

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        include_route = self.model_name == "catboost_route_rpm"
        include_statistics = self.route_statistics is not None
        if self.route_statistics is None:
            features = make_features(frame)
        else:
            features = add_route_statistics(frame, self.route_statistics)

        if self.model_name.startswith("catboost"):
            model_features = _catboost_features(
                features,
                include_route=include_route,
                include_route_statistics=include_statistics,
                include_market_quote=self.include_market_quote,
            )
        elif self.model_name == "huber_rpm":
            model_features = features[list(HUBER_NUMERIC_FEATURES + HUBER_CATEGORICAL_FEATURES)]
        else:
            model_features = features
        predictions = np.asarray(self.estimator.predict(model_features), dtype=float)
        if self.target_mode == "log_rpm":
            predictions = np.expm1(np.clip(predictions, -20, 20))
            distance = pd.to_numeric(frame["distance"], errors="coerce").clip(lower=1)
            predictions = predictions * distance.to_numpy(dtype=float)
        elif self.target_mode == "log_rpm_plain":
            predictions = np.exp(np.clip(predictions, -20, 20))
            distance = pd.to_numeric(frame["distance"], errors="coerce").clip(lower=1)
            predictions = predictions * distance.to_numpy(dtype=float)
        elif self.target_mode == "rpm":
            distance = pd.to_numeric(frame["distance"], errors="coerce")
            predictions = predictions * distance.to_numpy(dtype=float)
        return predictions

class BlendedModel:
    """I combine predictions from two fitted models."""

    def __init__(
        self,
        first: FreightModel,
        second: FreightModel,
        first_weight: float = 0.3,
    ) -> None:
        self.first = first
        self.second = second
        self.first_weight = first_weight

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        first_predictions = self.first.predict(frame)
        second_predictions = self.second.predict(frame)
        return self.first_weight * first_predictions + (1.0 - self.first_weight) * second_predictions

def fit_model(
    train_frame: pd.DataFrame,
    seed: int = SEED,
    model_name: str = "hist_gradient_boosting",
) -> FreightModel | BlendedModel:
    """I fit one candidate model on the labeled rows."""
    if TARGET not in train_frame.columns:
        raise ValueError(f"training data must contain {TARGET!r}")
    include_market_quote = not model_name.endswith("_no_market_quote")
    core_name = (
        model_name.removesuffix("_no_market_quote")
        if not include_market_quote
        else model_name
    )
    suffix = "" if include_market_quote else "_no_market_quote"
    if core_name == "blend":
        return BlendedModel(
            fit_model(train_frame, seed=seed, model_name=f"ridge{suffix}"),
            fit_model(
                train_frame,
                seed=seed,
                model_name=f"hist_gradient_boosting{suffix}",
            ),
        )
    if core_name == "catboost_hgb_rpm_blend":
        return BlendedModel(
            fit_model(train_frame, seed=seed, model_name=f"catboost{suffix}"),
            fit_model(
                train_frame,
                seed=seed,
                model_name=f"hist_gradient_boosting_rpm{suffix}",
            ),
            first_weight=0.30,
        )
    core_base_names = {
        name.removesuffix("_no_market_quote") for name in BASE_MODEL_NAMES
    }
    if core_name not in core_base_names:
        raise ValueError(f"unknown model {model_name!r}; choose from {MODEL_NAMES}")

    target = train_frame[TARGET].astype(float).to_numpy()
    is_rpm = core_name.endswith("_rpm") and core_name not in {
        "huber_rpm",
        "catboost_weighted_rpm",
    }
    target_mode = "log_rpm" if is_rpm else "raw"
    if is_rpm:
        distance = pd.to_numeric(train_frame["distance"], errors="coerce").clip(lower=1)
        target = np.log1p(target / distance.to_numpy(dtype=float))
    elif core_name == "huber_rpm":
        target_mode = "log_rpm_plain"
        distance = pd.to_numeric(train_frame["distance"], errors="coerce").clip(lower=1)
        target = np.log(target / distance.to_numpy(dtype=float))
    elif core_name == "catboost_weighted_rpm":
        target_mode = "rpm"
        distance = pd.to_numeric(train_frame["distance"], errors="coerce")
        target = target / distance.to_numpy(dtype=float)

    route_statistics = None
    if core_name == "catboost_route_rpm":
        route_statistics = fit_route_statistics(
            train_frame,
            target,
            smoothing=20.0,
            half_life_days=60.0,
        )
        features = add_route_statistics(
            train_frame, route_statistics, target=target, leave_one_out=True
        )
    else:
        features = make_features(train_frame)

    estimator = build_model(
        core_name,
        seed,
        include_market_quote=include_market_quote,
    )
    if core_name.startswith("catboost"):
        cat_features = ["pickup", "delivery", "equipment", "distance_bin"]
        if core_name == "catboost_route_rpm":
            cat_features.append("route_key")
        fit_features = _catboost_features(
                features,
                include_route=core_name == "catboost_route_rpm",
                include_route_statistics=route_statistics is not None,
                include_market_quote=include_market_quote,
            )
        if core_name == "catboost_weighted_rpm":
            sample_weight = pd.to_numeric(train_frame["distance"], errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            sample_weight = sample_weight / sample_weight.mean()
            estimator.fit(
                fit_features,
                target,
                sample_weight=sample_weight,
                cat_features=cat_features,
            )
        else:
            estimator.fit(fit_features, target, cat_features=cat_features)
    elif core_name == "huber_rpm":
        estimator.fit(features[list(HUBER_NUMERIC_FEATURES + HUBER_CATEGORICAL_FEATURES)], target)
    else:
        estimator.fit(features, target)
    return FreightModel(
        estimator,
        core_name,
        route_statistics,
        target_mode,
        include_market_quote,
    )

def predict_frame(model: FreightModel | BlendedModel, frame: pd.DataFrame) -> np.ndarray:
    """I predict rates and keep the results finite and positive."""
    predictions = np.asarray(model.predict(frame), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("model produced non-finite predictions")
    return np.maximum(predictions, 0.01)
