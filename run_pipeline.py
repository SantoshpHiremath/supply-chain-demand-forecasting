"""
End-to-end demo: generates data, trains the demand forecaster and the
delay-risk classifier, and produces a logistics recommendation (safety
stock / reorder point) for every part, printing an honest summary
including the naive-baseline comparison and the delay-risk metrics.
"""
import os

import pandas as pd

from src.generate_data import generate_demand_data, generate_shipment_data
from src.forecasting import forecast_all_parts
from src.delay_risk import train_delay_risk_model
from src.logistics_optimization import recommend_for_part, lead_time_stats_from_shipments


def main():
    os.makedirs("data", exist_ok=True)
    demand_df = generate_demand_data(seed=42)
    shipment_df = generate_shipment_data(seed=43)
    demand_df.to_csv("data/demand_history.csv", index=False)
    shipment_df.to_csv("data/shipment_history.csv", index=False)

    print("=" * 70)
    print("1. DEMAND FORECASTING (blended GBR + naive lag-1 baseline)")
    print("=" * 70)
    forecast_results = forecast_all_parts(demand_df, test_weeks=26)
    print(forecast_results.to_string(index=False))
    avg_improvement = forecast_results["improvement_over_naive_pct"].mean()
    wins = (forecast_results["improvement_over_naive_pct"] > 0).sum()
    print(f"\nAverage improvement over naive baseline: {avg_improvement:.1f}%")
    print(f"Parts where the blended model beats naive: {wins}/{len(forecast_results)}")
    print("(See README 'Honest finding' section for why this is modest, not dramatic.)")

    print("\n" + "=" * 70)
    print("2. DELIVERY DELAY-RISK CLASSIFICATION")
    print("=" * 70)
    pipeline, delay_metrics, _ = train_delay_risk_model(shipment_df)
    for key, value in delay_metrics.items():
        print(f"  {key}: {value:.3f}")

    print("\n" + "=" * 70)
    print("3. LOGISTICS RECOMMENDATIONS (safety stock / reorder point)")
    print("=" * 70)
    rows = []
    for part_id in sorted(demand_df["part_id"].unique()):
        part_demand = demand_df[demand_df.part_id == part_id]["demand_units"]
        avg_lt, std_lt = lead_time_stats_from_shipments(shipment_df, part_id)
        rec = recommend_for_part(
            part_id, part_demand.mean(), part_demand.std(), avg_lt, std_lt, service_level=0.95,
        )
        rows.append({
            "part_id": rec.part_id,
            "avg_weekly_demand": round(rec.avg_weekly_demand, 1),
            "avg_lead_time_weeks": round(rec.avg_lead_time_weeks, 2),
            "safety_stock_units": round(rec.safety_stock_units, 1),
            "reorder_point_units": round(rec.reorder_point_units, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
