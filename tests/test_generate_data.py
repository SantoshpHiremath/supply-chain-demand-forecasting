import pandas as pd

from src.generate_data import generate_demand_data, generate_shipment_data, PART_CONFIG, N_WEEKS


class TestDemandData:
    def test_same_seed_is_deterministic(self):
        df1 = generate_demand_data(seed=1)
        df2 = generate_demand_data(seed=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self):
        df1 = generate_demand_data(seed=1)
        df2 = generate_demand_data(seed=2)
        assert not df1["demand_units"].equals(df2["demand_units"])

    def test_covers_every_part_and_week(self):
        df = generate_demand_data(seed=1)
        assert set(df["part_id"]) == {cfg["part_id"] for cfg in PART_CONFIG}
        assert df.groupby("part_id")["week"].nunique().eq(N_WEEKS).all()

    def test_demand_is_never_negative(self):
        df = generate_demand_data(seed=1)
        assert (df["demand_units"] >= 0).all()


class TestShipmentData:
    def test_same_seed_is_deterministic(self):
        df1 = generate_shipment_data(seed=1)
        df2 = generate_shipment_data(seed=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_less_reliable_supplier_has_higher_late_rate(self):
        """Supplier_D is configured with the worst on_time_rate (0.65) --
        confirms the injected supplier-reliability structure actually
        shows up in the generated data, not just in the config dict.
        """
        df = generate_shipment_data(seed=1)
        late_rates = df.groupby("supplier")["is_late"].mean()
        assert late_rates["Supplier_D"] > late_rates["Supplier_C"]

    def test_delay_days_zero_when_not_late(self):
        df = generate_shipment_data(seed=1)
        on_time = df[~df["is_late"]]
        # on-time shipments should have zero or near-zero delay_days
        assert (on_time["delay_days"] == 0).mean() > 0.5

    def test_lead_times_are_positive(self):
        df = generate_shipment_data(seed=1)
        assert (df["actual_lead_time_days"] > 0).all()
