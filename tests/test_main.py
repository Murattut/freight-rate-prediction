from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from main import (
    add_route_statistics,
    build_run_manifest,
    canonicalize_categories,
    make_cold_start_split,
    build_error_case_audit,
    latest_two_month_split,
    enrich_chart_inputs,
    fit_route_statistics,
    fit_model,
    feature_columns,
    make_features,
    predict_frame,
    run_backtest,
    run_forecast_diagnostics,
    summarize_selected_errors,
    segment_metrics,
    temporal_splits,
    DEFAULT_MODEL_NAMES,
    EXTENDED_MODEL_NAMES,
    MODEL_NAMES,
    REJECTED_MODEL_NAMES,
    FreightModel,
    build_model,
    validate_data_contract,
    write_december_predictions,
    write_submission,
)


def _frame(months: int = 4, include_target: bool = True) -> pd.DataFrame:
    rows = []
    for month in range(1, months + 1):
        for i in range(3):
            rows.append(
                {
                    "load_id": f"ID-{month:02d}-{i}",
                    "pickup": "Alpha" if i != 2 else "Beta",
                    "delivery": "Gamma" if i != 1 else "Delta",
                    "pickup_lat": 35.0 + i,
                    "pickup_lon": -90.0 - i,
                    "delivery_lat": 36.0 + i,
                    "delivery_lon": -89.0 - i,
                    "distance": 300.0 + month * 25 + i,
                    "equipment": ["Dry Van", "Reefer", "Flatbed"][i],
                    "weight": [30000.0, -32000.0, np.nan][i],
                    "date": pd.Timestamp(2025, month, 1),
                    "market_index": [1.0, np.nan, 1.1][i],
                    "quote_signal": 2.0 + i / 10,
                    **({"posted_rate": 500.0 + month * 20 + i} if include_target else {}),
                }
            )
    return pd.DataFrame(rows)


def test_temporal_splits_are_forward_only() -> None:
    folds = list(temporal_splits(_frame(months=4)))

    assert len(folds) == 3
    for train, valid, train_end, valid_month in folds:
        assert train["date"].max() < valid["date"].min()
        assert train_end < valid_month


def test_forecast_diagnostics_use_frozen_cutoff_and_calendar_horizons() -> None:
    contiguous = run_forecast_diagnostics(
        _frame(months=6), model_name="ridge", min_train_months=4, seed=7
    )
    april = contiguous[contiguous.train_end == "2025-04"]
    assert april.valid_month.tolist() == ["2025-05", "2025-06"]
    assert april.horizon_months.tolist() == [1, 2]
    assert april.n_train.eq(12).all()
    assert april.evaluation_role.eq("development_diagnostic").all()

    with_gap = _frame(months=6)
    with_gap = with_gap[with_gap.date.dt.month != 5].copy()
    gap_metrics = run_forecast_diagnostics(
        with_gap, model_name="ridge", min_train_months=4, seed=7
    )
    gap_april = gap_metrics[gap_metrics.train_end == "2025-04"]
    assert gap_april.valid_month.tolist() == ["2025-06"]
    assert gap_april.horizon_months.tolist() == [2]


def test_latest_two_month_split_uses_calendar_months_and_requires_both() -> None:
    frame = _frame(months=6)
    train, future, cutoff = latest_two_month_split(frame, min_train_months=4)
    assert cutoff == "2025-04"
    assert train.date.dt.to_period("M").nunique() == 4
    assert future.date.dt.to_period("M").astype(str).unique().tolist() == ["2025-05", "2025-06"]
    assert future.date.dt.month.value_counts().tolist() == [3, 3]

    with_gap = frame[frame.date.dt.month != 5].copy()
    with pytest.raises(ValueError, match="no cutoff"):
        latest_two_month_split(with_gap, min_train_months=4)


def test_selected_oof_error_summary_reports_tail_share_and_empty_guard() -> None:
    oof = pd.DataFrame(
        {
            "model": ["selected"] * 100 + ["decoy"],
            "actual": [0.0] * 99 + [100.0, 1_000_000.0],
            "prediction": [0.0] * 101,
        }
    )
    summary = summarize_selected_errors(oof, selected_model="selected")

    assert summary["n"] == 100
    assert summary["mae"] == 1.0
    assert summary["pooled_rmse"] == 10.0
    assert summary["median_absolute_error"] == 0.0
    assert summary["p99_absolute_error"] == pytest.approx(1.0)
    assert summary["worst_1pct_squared_error_share"] == 1.0
    assert summarize_selected_errors(oof.iloc[0:0], selected_model="selected")["n"] == 0
    zeros = pd.DataFrame(
        {"model": ["selected", "selected"], "actual": [0.0, 0.0], "prediction": [0.0, 0.0]}
    )
    zero_summary = summarize_selected_errors(zeros, selected_model="selected")
    assert zero_summary["n"] == 2
    assert zero_summary["worst_1pct_squared_error_share"] == 0.0


def test_compact_default_models_include_robust_safe_candidate() -> None:
    assert set(DEFAULT_MODEL_NAMES) == {
        "ridge_no_market_quote",
        "catboost_no_market_quote",
        "catboost_weighted_rpm_no_market_quote",
    }
    assert set(DEFAULT_MODEL_NAMES).issubset(MODEL_NAMES)
    assert "huber_rpm_no_market_quote" in EXTENDED_MODEL_NAMES
    assert "huber_rpm_no_market_quote" in REJECTED_MODEL_NAMES
    metrics, selected, _ = run_backtest(
        _frame(months=5), seed=7, min_train_months=2, holdout_fold_count=1
    )
    assert set(metrics.model) == set(DEFAULT_MODEL_NAMES)
    assert selected in DEFAULT_MODEL_NAMES


def test_cold_start_split_is_deterministic_and_removes_city_overlap() -> None:
    train = _frame(months=4)
    future = _frame(months=5).copy()
    future["date"] = pd.Timestamp("2025-09-01")
    train["pickup"] = [f"Pickup {i % 6}" for i in range(len(train))]
    train["delivery"] = [f"Delivery {i % 6}" for i in range(len(train))]
    future["pickup"] = [f"Pickup {i % 6}" for i in range(len(future))]
    future["delivery"] = [f"Delivery {i % 6}" for i in range(len(future))]

    left = make_cold_start_split(train, future, seed=42, heldout_category_count=1)
    right = make_cold_start_split(train, future, seed=42, heldout_category_count=1)
    reduced_train, stress_rows, cities = left

    assert cities == right[2]
    assert len(stress_rows) > 0
    assert stress_rows.date.min() > train.date.max()
    normalized_reduced = canonicalize_categories(reduced_train)
    heldout_union = set(cities["all_categories"])
    assert not heldout_union & set(normalized_reduced.pickup)
    assert not heldout_union & set(normalized_reduced.delivery)
    normalized_stress = canonicalize_categories(stress_rows)
    assert normalized_stress.pickup.isin(heldout_union).any() or normalized_stress.delivery.isin(heldout_union).any()
    assert cities["n_train_before"] == len(train)
    assert cities["n_train_after"] == len(reduced_train)
    assert cities["n_train_removed"] == len(train) - len(reduced_train)
    assert cities["n_stress_rows"] == len(stress_rows)


def test_error_case_audit_keeps_positive_and_negative_extremes() -> None:
    train = _frame(months=1).iloc[:3].copy()
    oof = pd.DataFrame(
        {
            "model": ["huber_rpm_no_market_quote"] * 3,
            "load_id": train.load_id,
            "actual": [1000.0, 10.0, 500.0],
            "prediction": [100.0, 200.0, 300.0],
            "residual": [900.0, -190.0, 200.0],
        }
    )
    audit = build_error_case_audit(train, oof, selected_model="huber_rpm_no_market_quote")

    assert audit.case_type.tolist() == ["positive_residual", "positive_residual", "negative_residual"]
    assert audit.load_id.is_unique
    assert np.allclose(audit.actual_to_predicted_ratio, [10.0, 5.0 / 3.0, 0.05])
    expected_pickup = train.set_index("load_id").loc[audit.load_id, "pickup"].to_numpy()
    assert audit.pickup.tolist() == expected_pickup.tolist()


def test_huber_rate_per_mile_model_handles_missing_weights_and_unseen_categories() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False).copy()
    future["pickup"] = "Unseen pickup"
    future["delivery"] = "Unseen delivery"

    model = fit_model(train, seed=7, model_name="huber_rpm_no_market_quote")
    predictions = predict_frame(model, future)

    assert model.target_mode == "log_rpm_plain"
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_huber_plain_log_rate_inverse_multiplies_by_distance() -> None:
    class ConstantLogRate:
        def predict(self, features):
            return np.full(len(features), np.log(2.0))

    future = _frame(months=1, include_target=False)
    model = FreightModel(
        estimator=ConstantLogRate(),
        model_name="huber_rpm",
        target_mode="log_rpm_plain",
        include_market_quote=False,
    )

    np.testing.assert_allclose(model.predict(future), 2.0 * future.distance.to_numpy(dtype=float))


def test_weighted_rpm_model_inverse_multiplies_raw_rate_by_distance() -> None:
    class ConstantRatePerMile:
        def predict(self, features):
            return np.full(len(features), 2.0)

    future = _frame(months=1, include_target=False)
    future.loc[0, "distance"] = 0.5
    model = FreightModel(
        estimator=ConstantRatePerMile(),
        model_name="catboost_weighted_rpm",
        target_mode="rpm",
        include_market_quote=False,
    )
    np.testing.assert_allclose(
        model.predict(future), 2.0 * future.distance.to_numpy(dtype=float)
    )


def test_weighted_rpm_fit_receives_raw_rpm_targets_and_distance_weights(monkeypatch) -> None:
    class CapturingEstimator:
        def fit(self, features, target, *, sample_weight, cat_features):
            self.features = features
            self.target = np.asarray(target)
            self.sample_weight = np.asarray(sample_weight)
            self.cat_features = cat_features

    train = _frame(months=1).iloc[:3].copy()
    train["distance"] = [0.5, 4.0, 6.0]
    train["posted_rate"] = [5.0, 48.0, 84.0]
    original_train = train.copy(deep=True)
    estimator = CapturingEstimator()
    monkeypatch.setattr("freight_models.build_model", lambda *args, **kwargs: estimator)

    fit_model(train, model_name="catboost_weighted_rpm_no_market_quote")

    np.testing.assert_allclose(estimator.target, [10.0, 12.0, 14.0])
    np.testing.assert_allclose(estimator.sample_weight, [1.0 / 7.0, 8.0 / 7.0, 12.0 / 7.0])
    assert len(estimator.features) == 3
    pd.testing.assert_frame_equal(train, original_train)


def test_weighted_rpm_model_predicts_positive_finite_rates_for_unseen_categories() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False).copy()
    future["pickup"] = "Unseen pickup"
    future["delivery"] = "Unseen delivery"
    model = fit_model(train, seed=7, model_name="catboost_weighted_rpm_no_market_quote")
    predictions = predict_frame(model, future)

    assert model.target_mode == "rpm"
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_build_model_keeps_public_safe_suffix_compatibility() -> None:
    ridge = build_model("ridge_no_market_quote")
    huber = build_model("huber_rpm_no_market_quote")
    weighted = build_model("catboost_weighted_rpm_no_market_quote")

    assert ridge is not None
    assert huber is not None
    assert weighted is not None


def test_cleaning_marks_invalid_values_and_removes_target_leakage() -> None:
    features = make_features(_frame(months=1))

    assert "posted_rate" not in features.columns
    assert "load_id" not in features.columns
    assert features["weight_invalid"].sum() == 2
    assert features["weight_missing"].sum() == 1
    assert features["market_index_missing"].sum() == 1
    assert (features["weight_clean"].dropna() > 0).all()


def test_domain_features_include_distance_and_calendar_signals() -> None:
    features = make_features(_frame(months=1))

    assert {
        "distance_bin",
        "weight_per_mile",
        "route_key",
        "haversine_miles",
        "is_holiday",
    }.issubset(features.columns)
    assert features.loc[0, "route_key"] == "alpha -> gamma"
    assert features["weight_per_mile"].iloc[0] > 0
    assert np.isfinite(features["haversine_miles"]).all()


def test_canonical_categories_normalize_whitespace_and_case() -> None:
    frame = _frame(months=1)
    frame.loc[0, "pickup"] = "  ALPHA  "
    frame.loc[0, "delivery"] = "gamma"
    frame.loc[0, "equipment"] = "DRY VAN"

    canonical = canonicalize_categories(frame)

    assert canonical.loc[0, "pickup"] == "alpha"
    assert canonical.loc[0, "delivery"] == "gamma"
    assert canonical.loc[0, "equipment"] == "dry van"


def test_data_contract_rejects_duplicate_load_ids() -> None:
    frame = _frame(months=1)
    frame.loc[1, "load_id"] = frame.loc[0, "load_id"]

    with pytest.raises(ValueError, match="duplicate load_id"):
        validate_data_contract(frame, source="fixture", require_target=True)


def test_data_contract_allows_unverified_market_features_to_be_absent() -> None:
    frame = _frame(months=1).drop(columns=["market_index", "quote_signal"])

    validate_data_contract(frame, source="fixture", require_target=True)
    features = make_features(frame)

    assert features["market_index_missing"].eq(1).all()


def test_data_contract_rejects_non_numeric_optional_market_features() -> None:
    frame = _frame(months=1)
    frame["quote_signal"] = frame["quote_signal"].astype(object)
    frame.loc[0, "quote_signal"] = "not-a-number"

    with pytest.raises(ValueError, match="invalid numeric values in quote_signal"):
        validate_data_contract(frame, source="fixture", require_target=True)


def test_route_statistics_use_leave_one_out_values_for_training_rows() -> None:
    train = _frame(months=2)
    target = np.arange(len(train), dtype=float)
    statistics = fit_route_statistics(train, target)
    encoded = add_route_statistics(train, statistics, target=target, leave_one_out=True)

    same_route = train["pickup"].eq("Alpha") & train["delivery"].eq("Gamma")
    expected = target[same_route].mean()
    first_index = np.flatnonzero(same_route.to_numpy())[0]
    assert encoded.loc[first_index, "route_target_mean"] != expected
    assert encoded.loc[first_index, "route_target_mean"] == pytest.approx(
        target[same_route.to_numpy()][1:].mean()
    )


def test_catboost_rate_per_mile_model_predicts_positive_finite_rates() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False)

    model = fit_model(train, seed=7, model_name="catboost_rpm")
    predictions = predict_frame(model, future)

    assert len(predictions) == len(future)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_smoothed_route_model_predicts_positive_finite_rates() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False)

    model = fit_model(train, seed=7, model_name="catboost_route_rpm")
    predictions = predict_frame(model, future)

    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_backtest_returns_candidate_metrics_and_oof_predictions() -> None:
    train = _frame(months=4)
    metrics, best_model, oof = run_backtest(
        train,
        seed=7,
        min_train_months=2,
        model_names=("ridge", "catboost_rpm"),
        holdout_fold_count=1,
        market_quote_availability_verified=True,
    )

    assert best_model in {"ridge", "catboost_rpm"}
    assert {
        "model",
        "mae",
        "rmse",
        "n_train",
        "n_valid",
        "fold_role",
    }.issubset(metrics.columns)
    assert metrics.loc[metrics.valid_month == "2025-03", "fold_role"].eq("tune").all()
    assert metrics.loc[metrics.valid_month == "2025-04", "fold_role"].eq("holdout").all()
    tune_summary = (
        metrics.query("fold_role == 'tune'")
        .groupby("model")[["mae", "rmse"]]
        .mean()
        .sort_values(["mae", "rmse"])
    )
    assert best_model == tune_summary.index[0]
    assert {"model", "actual", "prediction", "residual"}.issubset(oof.columns)
    assert len(oof) == len(train[train["date"].dt.month >= 3]) * 2

    segments = segment_metrics(oof)
    assert {"model", "segment", "segment_value", "mae", "rmse", "n"}.issubset(
        segments.columns
    )
    assert len(segments) > 0


def test_run_manifest_records_data_and_selection_metadata() -> None:
    train = _frame(months=2)
    validation = _frame(months=1, include_target=False)
    metrics = pd.DataFrame(
        {
            "model": ["ridge_no_market_quote", "ridge_no_market_quote"],
            "mae": [1.0, 9.0],
            "rmse": [2.0, 10.0],
            "valid_month": ["2025-03", "2025-04"],
            "fold_role": ["tune", "holdout"],
        }
    )

    manifest = build_run_manifest(
        train,
        validation,
        metrics,
        selected_model="ridge_no_market_quote",
        seed=7,
    )

    assert manifest["selected_model"] == "ridge_no_market_quote"
    assert manifest["seed"] == 7
    assert manifest["train"]["rows"] == len(train)
    assert manifest["validation"]["rows"] == len(validation)
    assert manifest["metrics"]["best_mae"] == 1.0
    assert manifest["metrics"]["holdout_mae"] == 9.0
    assert manifest["evaluation_protocol"]["tuning_folds"] == ["2025-03"]
    assert manifest["evaluation_protocol"]["holdout_folds"] == ["2025-04"]
    assert manifest["evaluation_protocol"]["selection_uses_holdout_metrics"] is False
    assert manifest["feature_policy"]["market_index_and_quote_signal_source_timestamps_verified"] is False

    with_huber = pd.concat(
        [
            metrics,
            pd.DataFrame(
                {
                    "model": ["huber_rpm_no_market_quote"],
                    "mae": [0.5],
                    "rmse": [1.0],
                    "valid_month": ["2025-03"],
                    "fold_role": ["tune"],
                }
            ),
        ],
        ignore_index=True,
    )
    eligibility_manifest = build_run_manifest(
        train,
        validation,
        with_huber,
        selected_model="ridge_no_market_quote",
        seed=7,
    )
    assert "huber_rpm_no_market_quote" not in eligibility_manifest["selection_candidate_models"]
    assert "huber_rpm_no_market_quote" in eligibility_manifest["rejected_models"]
    assert eligibility_manifest["rejected_model_reasons"]["huber_rpm_no_market_quote"]


def test_backtest_selection_defaults_to_availability_safe_candidates(monkeypatch) -> None:
    class ConstantModel:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), self.value)

    monkeypatch.setattr(
        "main.fit_model",
        lambda train, seed, model_name: ConstantModel(
            540.0 if model_name == "ridge" else 1.0
        ),
    )
    metrics, best_model, _ = run_backtest(
        _frame(months=4),
        seed=7,
        min_train_months=2,
        model_names=("ridge", "ridge_no_market_quote", "huber_rpm_no_market_quote"),
        holdout_fold_count=1,
    )

    assert set(metrics["model"]) == {
        "ridge", "ridge_no_market_quote", "huber_rpm_no_market_quote"
    }
    assert best_model == "ridge_no_market_quote"


def test_availability_safe_feature_profile_excludes_market_inputs() -> None:
    numeric, categorical = feature_columns(include_market_quote=False)

    assert "market_index" not in numeric
    assert "market_index_missing" not in numeric
    assert "quote_signal" not in numeric
    default_numeric, _ = feature_columns()
    assert {"market_index", "quote_signal"}.issubset(default_numeric)


def test_availability_safe_blend_predicts_without_market_inputs() -> None:
    train = _frame(months=3).drop(columns=["market_index", "quote_signal"])
    future = _frame(months=1, include_target=False).drop(
        columns=["market_index", "quote_signal"]
    )

    model = fit_model(
        train,
        seed=7,
        model_name="catboost_hgb_rpm_blend_no_market_quote",
    )
    predictions = predict_frame(model, future)

    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_model_predicts_unseen_categories_with_positive_finite_rates() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False).copy()
    future["pickup"] = "Unseen pickup"
    future["delivery"] = "Unseen delivery"

    model = fit_model(train, seed=7, model_name="ridge")
    predictions = predict_frame(model, future)

    assert len(predictions) == len(future)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_model_predicts_sparse_december_inputs() -> None:
    train = _frame(months=3)
    december = pd.DataFrame(
        {
            "pickup": ["Unknown pickup"],
            "delivery": ["Unknown delivery"],
            "distance": [360.0],
            "equipment": ["Dry Van"],
            "weight": [32000.0],
            "date": [pd.Timestamp("2025-12-01")],
            "predicted_rate": [np.nan],
        }
    )

    model = fit_model(train, seed=7, model_name="ridge")
    enriched = enrich_chart_inputs(december, train)
    predictions = predict_frame(model, enriched)

    assert {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(
        enriched.columns
    )
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_chart_enrichment_tolerates_absent_optional_market_features() -> None:
    train = _frame(months=2).drop(columns=["market_index", "quote_signal"])
    december = pd.DataFrame(
        {
            "pickup": ["Alpha"],
            "delivery": ["Gamma"],
            "distance": [360.0],
            "equipment": ["Dry Van"],
            "weight": [32000.0],
            "date": [pd.Timestamp("2025-12-01")],
        }
    )

    enriched = enrich_chart_inputs(december, train)

    assert enriched["market_index"].isna().all()
    assert enriched["quote_signal"].isna().all()


def test_prediction_writers_preserve_required_schemas(tmp_path) -> None:
    template = pd.DataFrame({"load_id": ["TE-000001", "TE-000002"]})
    prediction_path = tmp_path / "validation_predictions.csv"
    write_submission(
        template, np.array([200.0, 100.0]), prediction_path,
        prediction_ids=["TE-000002", "TE-000001"],
    )
    submitted = pd.read_csv(prediction_path)

    assert submitted.columns.tolist() == ["load_id", "predicted_rate"]
    assert submitted["load_id"].tolist() == template["load_id"].tolist()
    assert submitted["predicted_rate"].tolist() == [100.0, 200.0]
    assert (submitted["predicted_rate"] > 0).all()

    december = pd.DataFrame(
        {
            "pickup": ["Lexington"],
            "delivery": ["Fort Wayne"],
            "distance": [360.0],
            "equipment": ["Dry Van"],
            "weight": [32000.0],
            "date": ["2025-12-01"],
            "predicted_rate": [np.nan],
        }
    )
    december_path = tmp_path / "december_predictions.csv"
    write_december_predictions(december, np.array([999.0]), december_path)
    written_december = pd.read_csv(december_path)

    assert written_december.columns.tolist() == december.columns.tolist()
    assert written_december["predicted_rate"].tolist() == [999.0]

    with pytest.raises(ValueError, match="missing or duplicate"):
        write_submission(
            pd.DataFrame({"load_id": [None]}), np.array([100.0]), tmp_path / "bad.csv",
            prediction_ids=[None],
        )

    with pytest.raises(ValueError, match="ID sets"):
        write_submission(
            template, np.array([100.0, 200.0]), tmp_path / "bad_ids.csv",
            prediction_ids=["TE-000002", "TE-000003"],
        )
    with pytest.raises(ValueError, match="prediction count"):
        write_submission(
            template, np.array([100.0]), tmp_path / "bad_count.csv",
            prediction_ids=["TE-000001", "TE-000002"],
        )
    with pytest.raises(ValueError, match="finite and positive"):
        write_submission(
            template, np.array([100.0, np.inf]), tmp_path / "bad_value.csv",
            prediction_ids=template["load_id"],
        )
    with pytest.raises(ValueError, match="duplicate load_id"):
        write_submission(
            template, np.array([100.0, 200.0]), tmp_path / "bad_duplicate.csv",
            prediction_ids=["TE-000001", "TE-000001"],
        )


def test_training_is_deterministic() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False)

    model_a = fit_model(train, seed=42, model_name="ridge")
    model_b = fit_model(train, seed=42, model_name="ridge")

    np.testing.assert_allclose(predict_frame(model_a, future), predict_frame(model_b, future))


def test_blended_model_predicts_positive_finite_rates() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False)

    model = fit_model(train, seed=42, model_name="blend")
    predictions = predict_frame(model, future)

    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_catboost_rate_blend_predicts_positive_finite_rates() -> None:
    train = _frame(months=3)
    future = _frame(months=1, include_target=False)

    model = fit_model(train, seed=42, model_name="catboost_hgb_rpm_blend")
    predictions = predict_frame(model, future)

    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
