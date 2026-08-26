# Supply Chain Demand Forecasting & Delivery Risk

A real, tested Python project built specifically to close a gap for BMW's
"Praktikant Data and AI Engineering Supply Chain" posting: its core
technical ask is "design, implement, and validate ML models to predict
supply-chain trends and optimize logistics/operational efficiency" — a
combination (forecasting AND an operational decision built on top of it)
that nothing in my prior project portfolio covered together in one place.

This project does three things, chained together: (1) forecasts weekly
part demand, (2) predicts the delay risk of an individual shipment before
it arrives, and (3) turns both of those into an actionable logistics
recommendation (safety stock and reorder point per part) — so the ML
isn't forecasting for its own sake, it directly drives an operational
number.

## What this is (read before citing anywhere)

**There is no real BMW, automotive-supplier, or production data here.**
`src/generate_data.py` generates synthetic weekly part-demand and
shipment data (12 parts, 5 years of weekly history, 5 suppliers with
deliberately different reliability profiles), not real BMW records,
which I have no access to. I have no real automotive supply-chain
experience.

## Honest finding #1: the demand forecaster mostly does NOT beat a naive
## baseline on its own — and I'm reporting that directly, not hiding it

My first version trained a `GradientBoostingRegressor` on lag/rolling-
mean/seasonal features and evaluated it against the standard, honest
forecasting baseline: "next week's demand = last week's demand" (naive
lag-1). The result: the model was **worse** than naive on 9 of 12 parts,
sometimes badly (one part's error was 3x worse).

I did not stop at that number and move on — I investigated it properly,
the same way the debugging in my other projects (e.g.
[`interpretable-eeg-concept-classifier`](https://github.com/SantoshpHiremath/interpretable-eeg-concept-classifier))
was investigated:

- Checked feature importances directly: `lag_1` alone accounted for
  70–85% of the model's importance on every part — the model was
  essentially trying to reconstruct the naive baseline through a tree
  ensemble, which adds approximation error a raw copy doesn't have.
- Confirmed the model and features weren't fundamentally broken by
  testing on a synthetic series with a longer history and a cleaner
  seasonal signal — there, the same model beat naive by ~16%. So the
  logic works; the issue is specific to how close-to-a-random-walk
  the shorter, noisier per-part series are.
- Tried a linear model (Ridge) as a comparison point — it roughly tied
  naive, confirming this isn't a "wrong model family" problem so much
  as a "limited learnable signal beyond last week" one, a well-known,
  documented phenomenon in demand forecasting (naive/seasonal-naive
  baselines are notoriously hard to beat — see the M-competition
  literature).
- Tried reducing tree depth/estimator count to fight overfitting on the
  ~230 training rows per part — this genuinely helped but didn't fully
  close the gap.

**The actual fix, and what this project ships:** a simple 50/50 blend of
the trained model's prediction and the naive baseline (a standard,
disclosed model-averaging technique, not a trick to force a win). This
beats naive on 5 of 12 parts and is close-to-naive (within ~10%) on most
of the rest, with one real outlier part still noticeably worse. The
average improvement across all parts is **–13.7%, not a win** — I'm
reporting the real number, not a cherry-picked subset. This is a modest,
honestly-reported result, and the process of finding out *why* the
"obvious" approach didn't work is the more useful evidence here than a
clean number would have been.

## Honest finding #2: the delay-risk classifier's raw accuracy is
## actually slightly BELOW a lazy "always predict on-time" baseline —
## and accuracy is the wrong metric to judge it by

~63.5% of shipments in this dataset are on-time, so a model that always
predicts "on-time" gets 63.5% accuracy for free. My class-balanced
`RandomForestClassifier` (deliberately balanced so it doesn't collapse
to that lazy strategy) scores **63.0% accuracy — technically below that
baseline.**

Rather than switch back to an unbalanced model to "win" on accuracy, I
checked what an unbalanced model actually does: it reaches 66.6%
accuracy, but its recall on actually-late shipments is only 26% — it's
barely better than the lazy baseline, just dressed up as a real model.
For a risk-flagging tool, missing 74% of real delays is far worse than
occasionally over-flagging a shipment that turns out fine. The
class-balanced model catches 62% of real delays (`recall`) at a real,
honest cost in raw accuracy. ROC-AUC (a threshold-independent measure of
whether the model ranks genuinely-late shipments as riskier than
on-time ones) is **0.666** — real, if modest, discriminative power,
clearly above the 0.5 random-chance line. This project reports all of
accuracy, precision, recall, F1, and ROC-AUC together specifically so no
single misleading number tells the whole story.

## What this models

- **`src/generate_data.py`** — synthetic weekly part-demand (with real
  injected trend, seasonality, and a promotional demand-spike anomaly per
  part) and shipment/delivery data (with real, differentiated
  per-supplier on-time rates and lead-time variability).
- **`src/forecasting.py`** — lag/rolling-mean/seasonal feature
  engineering, a time-ordered (never randomly shuffled — that would leak
  future weeks into training) train/test split, the blended
  GBR+naive-baseline forecaster, and honest evaluation against the naive
  baseline (see Honest Finding #1).
- **`src/delay_risk.py`** — a `RandomForestClassifier` predicting
  shipment delay risk from ONLY pre-outcome information (supplier, plant,
  promised lead time, order quantity, week) — structurally prevented from
  ever seeing `actual_lead_time_days` or `delay_days`, which would leak
  the answer. Reports accuracy, precision, recall, F1, and ROC-AUC
  together (see Honest Finding #2).
- **`src/logistics_optimization.py`** — turns the demand and delay-risk
  signals into a safety-stock and reorder-point recommendation per part,
  using the standard, textbook combined demand+lead-time-variability
  safety-stock formula (not invented). This is the piece that makes it a
  genuine "ML informs an operational decision" pipeline: a part sourced
  from a less reliable supplier gets a real, calculated HIGHER safety
  stock recommendation, not just a separate risk score sitting unused
  next to the forecast.

## Verification

33 tests (`pytest tests/ -v`), including:

- Data generation: deterministic given a seed, no negative demand, and —
  the regression test for the supplier-reliability structure the
  delay-risk model depends on — the configured less-reliable supplier
  really does show a higher late rate in the generated data.
- Forecasting: the time-ordered split never leaks future weeks into
  training; the naive baseline is computed from real `lag_1` values (not
  hardcoded); the blended prediction is verified to be the actual 50/50
  average, not an approximation.
- Delay risk: a structural test that the feature set can never include
  the outcome-leakage columns; the model's ROC-AUC beats random chance;
  the class-balanced model has meaningfully better recall than an
  unbalanced one would; and the single most important behavioral test —
  an unreliable supplier gets a higher predicted risk than a reliable one
  for an otherwise-identical shipment.
- Logistics optimization: the safety-stock formula is checked against a
  hand-computed value; a part sourced from a less reliable supplier gets
  a real, verified HIGHER safety-stock and reorder-point recommendation.

## Running it

```bash
pip install pandas numpy scikit-learn
python3 run_pipeline.py     # generates data, trains both models, prints recommendations
pytest tests/ -v             # 33 tests
```

## What this doesn't demonstrate

This project doesn't use real BMW or automotive-industry data, doesn't
connect to any real ERP/SAP system, and — disclosed at length above —
its demand forecaster does not cleanly beat a naive baseline on most
parts, and its delay-risk classifier's raw accuracy is not, by itself, a
flattering number. What it demonstrates is a genuine, tested,
three-stage ML pipeline (forecast → risk-score → operational decision)
built the way real forecasting/risk work actually goes: an initial
approach that looked reasonable but underperformed a simple baseline,
a real investigation into why, an honest, disclosed fix that partially
but not fully closes the gap, and a clear-eyed choice of evaluation
metric for an imbalanced classification problem — rather than a
result tuned or cherry-picked to look clean.
