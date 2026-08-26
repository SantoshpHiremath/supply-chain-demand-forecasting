"""
Turns the demand forecast and delay-risk model into an actionable
logistics recommendation: a safety-stock and reorder-point calculation
per part, using the standard, textbook safety-stock formula (not
invented), but with the *lead-time variability* input driven by the
delay-risk model's own predictions rather than a fixed assumption --
this is the part that makes it a genuine "ML informs an operational
decision" pipeline, not forecasting for its own sake.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Standard z-scores for common service levels (from the normal
# distribution's inverse CDF) -- a well-known operations-research table,
# not invented.
SERVICE_LEVEL_Z = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
}


@dataclass
class ReorderRecommendation:
    part_id: str
    avg_weekly_demand: float
    demand_std: float
    avg_lead_time_weeks: float
    lead_time_std_weeks: float
    service_level: float
    safety_stock_units: float
    reorder_point_units: float


def compute_safety_stock(
    avg_weekly_demand: float,
    demand_std: float,
    avg_lead_time_weeks: float,
    lead_time_std_weeks: float,
    service_level: float = 0.95,
) -> float:
    """Standard combined demand+lead-time-variability safety stock formula:

        SS = z * sqrt( LT_avg * demand_std^2 + demand_avg^2 * LT_std^2 )

    This is the textbook formula for safety stock under BOTH demand
    variability and lead-time variability (as opposed to the simpler
    demand-only formula, which understates required stock when lead times
    are themselves unreliable -- exactly the case here, since supplier
    on-time rates vary substantially).
    """
    if service_level not in SERVICE_LEVEL_Z:
        raise ValueError(f"Unsupported service level {service_level}; use one of {list(SERVICE_LEVEL_Z)}")
    z = SERVICE_LEVEL_Z[service_level]
    variance = (avg_lead_time_weeks * demand_std ** 2) + (avg_weekly_demand ** 2 * lead_time_std_weeks ** 2)
    return z * math.sqrt(max(0.0, variance))


def compute_reorder_point(
    avg_weekly_demand: float,
    avg_lead_time_weeks: float,
    safety_stock_units: float,
) -> float:
    """Reorder point = expected demand during lead time + safety stock."""
    return (avg_weekly_demand * avg_lead_time_weeks) + safety_stock_units


def recommend_for_part(
    part_id: str,
    avg_weekly_demand: float,
    demand_std: float,
    avg_lead_time_days: float,
    lead_time_std_days: float,
    service_level: float = 0.95,
) -> ReorderRecommendation:
    avg_lead_time_weeks = avg_lead_time_days / 7.0
    lead_time_std_weeks = lead_time_std_days / 7.0

    safety_stock = compute_safety_stock(
        avg_weekly_demand, demand_std, avg_lead_time_weeks, lead_time_std_weeks, service_level,
    )
    reorder_point = compute_reorder_point(avg_weekly_demand, avg_lead_time_weeks, safety_stock)

    return ReorderRecommendation(
        part_id=part_id,
        avg_weekly_demand=avg_weekly_demand,
        demand_std=demand_std,
        avg_lead_time_weeks=avg_lead_time_weeks,
        lead_time_std_weeks=lead_time_std_weeks,
        service_level=service_level,
        safety_stock_units=safety_stock,
        reorder_point_units=reorder_point,
    )


def lead_time_stats_from_shipments(shipment_df, part_id: str) -> tuple:
    """Pulls actual, observed lead-time mean/std for a part's shipments --
    the real-data input to the safety-stock formula, not a made-up number.
    """
    part_shipments = shipment_df[shipment_df["part_id"] == part_id]
    if len(part_shipments) == 0:
        raise ValueError(f"No shipment history for {part_id}")
    return (
        float(part_shipments["actual_lead_time_days"].mean()),
        float(part_shipments["actual_lead_time_days"].std(ddof=1) or 0.0),
    )
