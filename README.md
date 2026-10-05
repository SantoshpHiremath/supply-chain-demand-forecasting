# Supply Chain Demand Forecasting & Delivery Risk

A tested Python project that forecasts weekly part demand, predicts shipment delay risk, and turns both into an operational logistics recommendation. It combines forecasting and an operational decision built on top of it: designing, implementing, and validating ML models to predict supply-chain trends and optimize logistics and operational efficiency.

## What it does

The project chains three steps: (1) forecasts weekly part demand, (2) predicts the delay risk of an individual shipment before it arrives, and (3) turns both into an actionable logistics recommendation (safety stock and reorder point per part), so the ML directly drives an operational number.

- **`src/generate_data.py`**: synthetic weekly part-demand (with injected trend, seasonality, and a promotional demand-spike anomaly per part) and shipment/delivery data (with differentiated per-supplier on-time rates and lead-time variability).
- **`src/forecasting.py`**: lag/rolling-mean/seasonal feature engineering, a time-ordered (never randomly shuffled, which would leak future weeks into training) train/test split, the blended GBR+naive-baseline forecaster, and evaluation against the naive baseline (see Results).
- **`src/delay_risk.py`**: a `RandomForestClassifier` predicting shipment delay risk from only pre-outcome information (supplier, plant, promised lead time, order quantity, week), structurally prevented from seeing `actual_lead_time_days` or `delay_days`, which would leak the answer. It reports accuracy, precision, recall, F1, and ROC-AUC together (see Results).
- **`src/logistics_optimization.py`**: turns the demand and delay-risk signals into a safety-stock and reorder-point recommendation per part, using the standard textbook combined demand and lead-time-variability safety-stock formula. A part sourced from a less reliable supplier gets a higher calculated safety stock recommendation, so the risk signal feeds the operational decision directly.

## Data

`src/generate_data.py` generates synthetic weekly part-demand and shipment data (12 parts, 5 years of weekly history, 5 suppliers with deliberately different reliability profiles). The pipeline is built so real ERP or supply-chain records can replace it.

## Results

### Demand forecasting

I trained a `GradientBoostingRegressor` on lag/rolling-mean/seasonal features and evaluated it against the standard forecasting baseline, "next week's demand = last week's demand" (naive lag-1). The first version was worse than naive on 9 of 12 parts, sometimes by a wide margin (one part's error was 3x worse), so I investigated:

- Feature importances showed `lag_1` alone accounted for 70-85% of the model's importance on every part, so the model was largely reconstructing the naive baseline through a tree ensemble, which adds approximation error a raw copy does not have.
- On a synthetic series with a longer history and a cleaner seasonal signal, the same model beat naive by ~16%, confirming the model and features work; the shorter, noisier per-part series sit close to a random walk.
- A linear model (Ridge) roughly tied naive, which points to limited learnable signal beyond last week rather than a wrong model family. This is a well-documented phenomenon in demand forecasting (naive and seasonal-naive baselines are notoriously hard to beat; see the M-competition literature).
- Reducing tree depth and estimator count to limit overfitting on the ~230 training rows per part helped but did not fully close the gap.

**What the project ships:** a simple 50/50 blend of the trained model's prediction and the naive baseline, a standard model-averaging technique. It beats naive on 5 of 12 parts and is close to naive (within ~10%) on most of the rest, with one outlier part still noticeably worse. The average improvement across all parts is **-13.7%** relative to naive.

### Delay-risk classification

~63.5% of shipments in this dataset are on-time, so a model that always predicts "on-time" reaches 63.5% accuracy. My class-balanced `RandomForestClassifier` (balanced so it does not collapse to that strategy) scores **63.0% accuracy**, and accuracy is not the right metric for this task:

- An unbalanced model reaches 66.6% accuracy, but its recall on actually-late shipments is only 26%, barely better than the always-on-time baseline.
- For a risk-flagging tool, missing 74% of real delays is far worse than occasionally over-flagging a shipment that turns out fine. The class-balanced model catches 62% of real delays (`recall`).
- ROC-AUC, a threshold-independent measure of whether the model ranks late shipments as riskier than on-time ones, is **0.666**, clearly above the 0.5 random-chance line.

The project reports accuracy, precision, recall, F1, and ROC-AUC together so the full picture is visible.

## Tests

33 tests (`pytest tests/ -v`), including:

- Data generation: deterministic given a seed, no negative demand, and (the regression test for the supplier-reliability structure the delay-risk model depends on) the configured less-reliable supplier shows a higher late rate in the generated data.
- Forecasting: the time-ordered split never leaks future weeks into training; the naive baseline is computed from real `lag_1` values (not hardcoded); the blended prediction is verified to be the actual 50/50 average.
- Delay risk: a structural test that the feature set can never include the outcome-leakage columns; the model's ROC-AUC beats random chance; the class-balanced model has meaningfully better recall than an unbalanced one would; and an unreliable supplier gets a higher predicted risk than a reliable one for an otherwise identical shipment.
- Logistics optimization: the safety-stock formula is checked against a hand-computed value; a part sourced from a less reliable supplier gets a higher safety-stock and reorder-point recommendation.

## Project structure

```
run_pipeline.py                 end-to-end runner
src/generate_data.py            synthetic demand and shipment data
src/forecasting.py              demand forecaster
src/delay_risk.py               delay-risk classifier
src/logistics_optimization.py   safety stock and reorder point
data/                           generated CSVs
tests/                          pytest suite
```

## Running it

```bash
pip install pandas numpy scikit-learn
python3 run_pipeline.py     # generates data, trains both models, prints recommendations
pytest tests/ -v             # 33 tests
```

## Notes

The demand forecaster ships as a blend with the naive baseline because the naive lag-1 baseline is strong on these series. The delay-risk model is tuned for recall on late shipments, which is the operationally important class.

## Possible extensions

- Connect to real ERP/SAP supply-chain data.
- Try seasonal-naive baselines and hierarchical forecasting across parts.
- Add probability calibration to the delay-risk model.
