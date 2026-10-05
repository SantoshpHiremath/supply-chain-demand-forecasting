"""
Shipment delay-risk classification: predicts, at order time, whether a
shipment is likely to arrive late, using only information available
BEFORE the shipment arrives (supplier, promised lead time, order quantity,
week-of-year) -- never the actual outcome fields (actual_lead_time_days,
delay_days), which would leak the answer.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Only pre-shipment-outcome information -- see module docstring on leakage.
FEATURE_COLS = ["supplier", "plant", "promised_lead_time_days", "order_qty", "week_of_year"]
CATEGORICAL_COLS = ["supplier", "plant"]
NUMERIC_COLS = ["promised_lead_time_days", "order_qty", "week_of_year"]


def build_features(shipment_df: pd.DataFrame) -> pd.DataFrame:
    df = shipment_df.copy()
    df["week_of_year"] = df["week"] % 52
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ], remainder="passthrough")
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced",
        )),
    ])


def train_delay_risk_model(shipment_df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    df = build_features(shipment_df)
    X = df[FEATURE_COLS]
    y = df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, list(pipeline.named_steps["classifier"].classes_).index(True)]
    majority_baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    # Accuracy alone is a misleading metric here -- see README's "Results"
    # section: the class-balanced model trades some raw accuracy
    # (which a lazy "always predict on-time" model can already win, since
    # ~63% of shipments genuinely are on time) for far better recall on
    # the operationally important minority class (actually-late
    # shipments). ROC-AUC (threshold-independent) and recall are the
    # metrics that actually reflect whether this model is useful for
    # flagging real delay risk.
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "majority_class_baseline_accuracy": majority_baseline_acc,
    }
    return pipeline, metrics, (X_test, y_test, y_pred)


def predict_delay_risk(pipeline: Pipeline, shipment_row: dict) -> float:
    """Returns the model's predicted probability that a single new
    shipment (described by the pre-outcome features only) will be late.
    """
    row_df = pd.DataFrame([shipment_row])[FEATURE_COLS]
    proba = pipeline.predict_proba(row_df)[0]
    # class order from the fitted classifier -- look up index of True
    classes = list(pipeline.named_steps["classifier"].classes_)
    return float(proba[classes.index(True)])
