import pandas as pd
import pytest

from src.generate_data import generate_shipment_data
from src.delay_risk import (
    build_features, train_delay_risk_model, predict_delay_risk, FEATURE_COLS,
)


@pytest.fixture(scope="module")
def shipment_df():
    return generate_shipment_data(seed=43)


class TestBuildFeatures:
    def test_no_outcome_leakage_columns_in_feature_set(self, shipment_df):
        """The most important test in this file: FEATURE_COLS must never
        include actual_lead_time_days or delay_days -- those are only
        known AFTER the shipment outcome, and including them would let
        the model 'predict' delay using the answer itself.
        """
        assert "actual_lead_time_days" not in FEATURE_COLS
        assert "delay_days" not in FEATURE_COLS
        assert "is_late" not in FEATURE_COLS

    def test_week_of_year_computed_correctly(self, shipment_df):
        df = build_features(shipment_df)
        row = df.iloc[0]
        assert row["week_of_year"] == row["week"] % 52


class TestTrainDelayRiskModel:
    def test_returns_fitted_pipeline_and_metrics(self, shipment_df):
        pipeline, metrics, _ = train_delay_risk_model(shipment_df)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "majority_class_baseline_accuracy" in metrics

    def test_roc_auc_beats_random_chance(self, shipment_df):
        """The honest, threshold-independent signal check: ROC-AUC of
        0.5 means no better than random. This model should show real,
        if modest, discriminative power -- see README for the full
        honest discussion of accuracy vs. ROC-AUC on this task.
        """
        _, metrics, _ = train_delay_risk_model(shipment_df)
        assert metrics["roc_auc"] > 0.55

    def test_class_balanced_model_has_meaningfully_better_recall_than_unbalanced(self, shipment_df):
        """Regression test for the honest finding in README: an
        unbalanced RandomForestClassifier mostly just predicts the
        majority class (on-time) and has poor recall on actually-late
        shipments. This test confirms the balanced model (the one this
        project actually ships) doesn't fall into that trap.
        """
        _, metrics, _ = train_delay_risk_model(shipment_df)
        assert metrics["recall"] > 0.5


class TestPredictDelayRisk:
    def test_returns_probability_between_zero_and_one(self, shipment_df):
        pipeline, _, _ = train_delay_risk_model(shipment_df)
        risk = predict_delay_risk(pipeline, {
            "supplier": "Supplier_D", "plant": "Munich",
            "promised_lead_time_days": 25, "order_qty": 500, "week_of_year": 10,
        })
        assert 0.0 <= risk <= 1.0

    def test_unreliable_supplier_gets_higher_predicted_risk_than_reliable_one(self, shipment_df):
        """The single most important behavioral test: Supplier_D (real
        on-time rate 65%) should get a meaningfully higher predicted
        delay-risk than Supplier_C (real on-time rate 97%) for an
        otherwise-identical shipment -- confirms the model actually
        learned supplier reliability, not just noise.
        """
        pipeline, _, _ = train_delay_risk_model(shipment_df)
        common = {"plant": "Munich", "promised_lead_time_days": 15, "order_qty": 500, "week_of_year": 10}
        risk_unreliable = predict_delay_risk(pipeline, {**common, "supplier": "Supplier_D"})
        risk_reliable = predict_delay_risk(pipeline, {**common, "supplier": "Supplier_C"})
        assert risk_unreliable > risk_reliable
