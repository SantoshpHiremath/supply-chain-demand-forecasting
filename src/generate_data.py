"""
Generates a synthetic but realistic multi-part automotive supply-chain
dataset: weekly part demand at a plant, supplier lead times, and shipment
delivery outcomes — built to give the forecasting and delay-risk models
something real to learn from, including deliberately injected realistic
structure (seasonality, trend, supplier-specific delay risk) rather than
pure noise.

Synthetic data. Modeled on the domain of material ordering at scale,
supply-chain risk assessment, and logistics optimization.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List

import numpy as np
import pandas as pd

N_PARTS = 12
N_WEEKS = 260  # 5 years of weekly history -- enough history per part for
                # lag/seasonal features to have real signal to learn from
PLANTS = ["Munich", "Regensburg", "Leipzig", "Dingolfing"]
SUPPLIERS = ["Supplier_A", "Supplier_B", "Supplier_C", "Supplier_D", "Supplier_E"]

# Each part has a base weekly demand, a trend, a seasonal amplitude, and an
# assigned primary supplier -- deliberately varied so the forecasting task
# has genuine part-to-part heterogeneity, not one repeated pattern.
PART_CONFIG = [
    {"part_id": f"PART-{i:03d}", "base_demand": base, "trend_per_week": trend,
     "seasonal_amp": seas, "plant": PLANTS[i % len(PLANTS)],
     "supplier": SUPPLIERS[i % len(SUPPLIERS)]}
    for i, (base, trend, seas) in enumerate([
        # seasonal_amp kept at a meaningful fraction of base_demand for
        # every part (roughly 15-20%), so the seasonal signal is
        # realistically learnable rather than swamped by noise on the
        # smaller-volume parts.
        (400, 0.8, 70), (150, -0.3, 25), (900, 1.5, 160), (60, 0.05, 10),
        (320, 0.4, 55), (210, -0.6, 35), (700, 1.0, 120), (40, 0.02, 7),
        (500, 0.6, 85), (180, 0.3, 30), (95, -0.1, 16), (260, 0.5, 45),
    ])
]

# Supplier-specific baseline on-time delivery rate and lead-time variability
# -- some suppliers are simply less reliable, a real and common supply-chain
# risk pattern this project's delay-risk model has to learn to distinguish.
SUPPLIER_PROFILE = {
    "Supplier_A": {"on_time_rate": 0.94, "base_lead_time_days": 12, "lead_time_std": 1.5},
    "Supplier_B": {"on_time_rate": 0.80, "base_lead_time_days": 18, "lead_time_std": 4.0},
    "Supplier_C": {"on_time_rate": 0.97, "base_lead_time_days": 9, "lead_time_std": 1.0},
    "Supplier_D": {"on_time_rate": 0.65, "base_lead_time_days": 25, "lead_time_std": 6.5},
    "Supplier_E": {"on_time_rate": 0.88, "base_lead_time_days": 15, "lead_time_std": 2.5},
}


def _seasonal_component(week: int, amplitude: float) -> float:
    """A yearly seasonal cycle (52-week period) plus a smaller quarterly
    wobble, so the seasonality isn't a single trivial sine wave."""
    yearly = amplitude * math.sin(2 * math.pi * week / 52.0)
    quarterly = (amplitude * 0.3) * math.sin(2 * math.pi * week / 13.0)
    return yearly + quarterly


def generate_demand_data(seed: int = 42) -> pd.DataFrame:
    """Weekly demand (units ordered) per part, with trend + seasonality +
    noise, plus a few realistic anomalies (promotion-driven demand spikes,
    a supply-shortage-driven demand dip) so the data isn't purely
    well-behaved.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cfg in PART_CONFIG:
        for week in range(N_WEEKS):
            trend = cfg["trend_per_week"] * week
            seasonal = _seasonal_component(week, cfg["seasonal_amp"])
            # Noise scaled to be smaller than the seasonal amplitude so the
            # seasonal/trend signal is actually learnable rather than
            # swamped -- weekly aggregated automotive-parts demand is
            # smoother than per-unit noise would suggest, since it sums
            # many independent vehicle-build orders.
            noise = rng.normal(0, cfg["base_demand"] * 0.035)
            demand = cfg["base_demand"] + trend + seasonal + noise

            # Deliberate anomaly: a demand spike in a randomly chosen
            # 3-week promotional window per part (real, not hidden).
            if cfg.get("promo_week") is None:
                cfg["promo_week"] = int(rng.integers(20, N_WEEKS - 20))
            if cfg["promo_week"] <= week < cfg["promo_week"] + 3:
                demand *= 1.6

            demand = max(0.0, demand)
            rows.append({
                "part_id": cfg["part_id"],
                "plant": cfg["plant"],
                "supplier": cfg["supplier"],
                "week": week,
                "demand_units": round(demand),
            })
    return pd.DataFrame(rows)


def generate_shipment_data(seed: int = 43) -> pd.DataFrame:
    """One row per shipment (a part/supplier order arriving in a given
    week), with a promised lead time and an actual delivery outcome
    (on-time / late, and how late) driven by the supplier's real
    reliability profile plus general noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    shipment_id = 0
    for cfg in PART_CONFIG:
        profile = SUPPLIER_PROFILE[cfg["supplier"]]
        for week in range(N_WEEKS):
            # Not every part has a shipment every week -- roughly biweekly
            # ordering cadence, which is realistic for automotive parts.
            if rng.random() > 0.5:
                continue
            shipment_id += 1
            promised_lead_time = profile["base_lead_time_days"]
            actual_lead_time = max(
                1.0,
                rng.normal(profile["base_lead_time_days"], profile["lead_time_std"]),
            )
            is_late = actual_lead_time > promised_lead_time * 1.1
            # Baseline on-time rate is enforced probabilistically too, so
            # the label isn't purely derived from the lead-time draw alone.
            if rng.random() > profile["on_time_rate"]:
                is_late = True
                actual_lead_time = max(actual_lead_time, promised_lead_time * 1.3)

            order_qty = rng.integers(50, 2000)
            rows.append({
                "shipment_id": shipment_id,
                "part_id": cfg["part_id"],
                "supplier": cfg["supplier"],
                "plant": cfg["plant"],
                "week": week,
                "order_qty": int(order_qty),
                "promised_lead_time_days": promised_lead_time,
                "actual_lead_time_days": round(actual_lead_time, 1),
                "is_late": bool(is_late),
                "delay_days": round(max(0.0, actual_lead_time - promised_lead_time), 1),
            })
    return pd.DataFrame(rows)


def main():
    demand_df = generate_demand_data()
    shipment_df = generate_shipment_data()
    demand_df.to_csv("data/demand_history.csv", index=False)
    shipment_df.to_csv("data/shipment_history.csv", index=False)
    print(f"Generated {len(demand_df)} demand rows across {len(PART_CONFIG)} parts, "
          f"{N_WEEKS} weeks.")
    print(f"Generated {len(shipment_df)} shipment rows. "
          f"Late rate: {shipment_df['is_late'].mean():.1%}")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()
