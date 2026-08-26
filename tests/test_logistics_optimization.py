import math

import pandas as pd
import pytest

from src.generate_data import generate_shipment_data
from src.logistics_optimization import (
    compute_safety_stock, compute_reorder_point, recommend_for_part,
    lead_time_stats_from_shipments, SERVICE_LEVEL_Z,
)


class TestComputeSafetyStock:
    def test_zero_variability_gives_zero_safety_stock(self):
        """If both demand and lead time are perfectly predictable (zero
        std), no safety stock is needed -- a basic sanity check on the
        formula.
        """
        ss = compute_safety_stock(
            avg_weekly_demand=100, demand_std=0,
            avg_lead_time_weeks=2, lead_time_std_weeks=0,
        )
        assert ss == pytest.approx(0.0)

    def test_higher_service_level_requires_more_safety_stock(self):
        ss_90 = compute_safety_stock(100, 20, 2, 0.5, service_level=0.90)
        ss_99 = compute_safety_stock(100, 20, 2, 0.5, service_level=0.99)
        assert ss_99 > ss_90

    def test_matches_hand_computed_formula(self):
        """Regression test against the textbook formula computed by hand,
        so a future refactor can't silently change the math.
        """
        z = SERVICE_LEVEL_Z[0.95]
        avg_demand, demand_std, avg_lt, lt_std = 100.0, 15.0, 2.0, 0.3
        expected = z * math.sqrt(avg_lt * demand_std**2 + avg_demand**2 * lt_std**2)
        actual = compute_safety_stock(avg_demand, demand_std, avg_lt, lt_std, service_level=0.95)
        assert actual == pytest.approx(expected)

    def test_rejects_unsupported_service_level(self):
        with pytest.raises(ValueError):
            compute_safety_stock(100, 20, 2, 0.5, service_level=0.80)


class TestComputeReorderPoint:
    def test_reorder_point_is_lead_time_demand_plus_safety_stock(self):
        rp = compute_reorder_point(avg_weekly_demand=50, avg_lead_time_weeks=3, safety_stock_units=40)
        assert rp == pytest.approx(50 * 3 + 40)


class TestRecommendForPart:
    def test_higher_lead_time_variability_increases_safety_stock(self):
        """The core 'ML informs an operational decision' claim: a part
        sourced from a less reliable supplier (higher lead-time std)
        should get a HIGHER safety-stock recommendation, all else equal
        -- confirms the delay-risk signal actually flows into the
        logistics recommendation, not just sitting unused alongside it.
        """
        rec_reliable = recommend_for_part(
            "PART-X", avg_weekly_demand=200, demand_std=30,
            avg_lead_time_days=14, lead_time_std_days=1.0,
        )
        rec_unreliable = recommend_for_part(
            "PART-X", avg_weekly_demand=200, demand_std=30,
            avg_lead_time_days=14, lead_time_std_days=6.0,
        )
        assert rec_unreliable.safety_stock_units > rec_reliable.safety_stock_units
        assert rec_unreliable.reorder_point_units > rec_reliable.reorder_point_units


class TestLeadTimeStatsFromShipments:
    def test_raises_for_unknown_part(self):
        shipment_df = generate_shipment_data(seed=43)
        with pytest.raises(ValueError):
            lead_time_stats_from_shipments(shipment_df, "PART-DOES-NOT-EXIST")

    def test_stats_match_manual_groupby(self):
        shipment_df = generate_shipment_data(seed=43)
        avg_lt, std_lt = lead_time_stats_from_shipments(shipment_df, "PART-000")
        subset = shipment_df[shipment_df.part_id == "PART-000"]["actual_lead_time_days"]
        assert avg_lt == pytest.approx(subset.mean())
        assert std_lt == pytest.approx(subset.std(ddof=1))
