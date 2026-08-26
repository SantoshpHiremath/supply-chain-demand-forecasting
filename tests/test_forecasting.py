import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_demand_data
from src.forecasting import (
    build_features, train_test_split_by_time, train_demand_forecaster,
    evaluate_forecaster, forecast_all_parts, FEATURE_COLS,
)


@pytest.fixture(scope="module")
def demand_df():
    return generate_demand_data(seed=42)


class TestBuildFeatures:
    def test_drops_rows_without_enough_lag_history(self, demand_df):
        feat_df = build_features(demand_df, "PART-000")
        # first 4 weeks can't have a lag_4 value
        assert feat_df["week"].min() >= 4

    def test_no_nans_in_feature_columns(self, demand_df):
        feat_df = build_features(demand_df, "PART-000")
        assert not feat_df[FEATURE_COLS].isna().any().any()

    def test_lag_1_matches_previous_week_demand(self, demand_df):
        feat_df = build_features(demand_df, "PART-000")
        part_df = demand_df[demand_df.part_id == "PART-000"].sort_values("week").reset_index(drop=True)
        row = feat_df.iloc[10]
        prev_week_demand = part_df[part_df.week == row["week"] - 1]["demand_units"].iloc[0]
        assert row["lag_1"] == prev_week_demand


class TestTimeOrderedSplit:
    def test_split_preserves_time_order_no_shuffle(self, demand_df):
        feat_df = build_features(demand_df, "PART-000")
        train_df, test_df = train_test_split_by_time(feat_df, test_weeks=26)
        assert train_df["week"].max() < test_df["week"].min()

    def test_test_set_has_requested_size(self, demand_df):
        feat_df = build_features(demand_df, "PART-000")
        train_df, test_df = train_test_split_by_time(feat_df, test_weeks=26)
        assert len(test_df) == 26


class TestForecaster:
    def test_model_predictions_are_reasonable_magnitude(self, demand_df):
        """Predictions shouldn't be wildly outside the training demand
        range -- a basic sanity check that the model learned something,
        not nonsense.
        """
        feat_df = build_features(demand_df, "PART-002")
        train_df, test_df = train_test_split_by_time(feat_df, test_weeks=26)
        model = train_demand_forecaster(train_df)
        preds = model.predict(test_df[FEATURE_COLS])
        train_min, train_max = train_df["demand_units"].min(), train_df["demand_units"].max()
        margin = (train_max - train_min) * 0.5
        assert preds.min() > train_min - margin
        assert preds.max() < train_max + margin

    def test_evaluate_reports_naive_baseline_honestly(self, demand_df):
        """Regression test for the honest-reporting finding in README:
        the naive baseline MUST be computed from actual lag_1 values, not
        hardcoded or fudged, so the comparison is trustworthy.
        """
        feat_df = build_features(demand_df, "PART-000")
        train_df, test_df = train_test_split_by_time(feat_df, test_weeks=26)
        model = train_demand_forecaster(train_df)
        metrics = evaluate_forecaster(model, test_df, blend_with_naive=False)
        expected_naive_mae = np.mean(np.abs(test_df["demand_units"].values - test_df["lag_1"].values))
        assert metrics["naive_baseline_mae"] == pytest.approx(expected_naive_mae)

    def test_blended_prediction_is_average_of_model_and_naive(self, demand_df):
        feat_df = build_features(demand_df, "PART-000")
        train_df, test_df = train_test_split_by_time(feat_df, test_weeks=26)
        model = train_demand_forecaster(train_df)
        raw_preds = model.predict(test_df[FEATURE_COLS])
        metrics = evaluate_forecaster(model, test_df, blend_with_naive=True)
        expected_blend = 0.5 * raw_preds + 0.5 * test_df["lag_1"].values
        np.testing.assert_allclose(metrics["predictions"], expected_blend)


class TestForecastAllParts:
    def test_returns_one_row_per_part_with_enough_history(self, demand_df):
        results = forecast_all_parts(demand_df, test_weeks=26)
        assert len(results) == demand_df["part_id"].nunique()

    def test_mae_values_are_non_negative(self, demand_df):
        results = forecast_all_parts(demand_df, test_weeks=26)
        assert (results["model_mae"] >= 0).all()
        assert (results["naive_baseline_mae"] >= 0).all()
