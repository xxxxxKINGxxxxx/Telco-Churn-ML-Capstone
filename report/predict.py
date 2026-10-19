#!/usr/bin/env python3
"""
predict.py — score customers with the trained Telco churn-risk pipeline (model.joblib).

The pipeline bundles preprocessing (scaling + one-hot encoding) and the tuned Random Forest
classifier together (see notebooks/telco_churn_analysis.ipynb, Section 6.1), so this script
only needs to load one file and pass it raw, unprocessed customer rows. It independently
reproduces the notebook's feature engineering (Section 3.3: `services_count`, `is_new_customer`)
and cleaning (Section 3.1) so it works standalone, without importing the notebook.

Usage
-----
Score every row in a CSV of raw customer records (same columns as the training data, minus
the target `Churn` and identifier `customerID`):

    python3 predict.py --input customers.csv --output scored_customers.csv

Score a single customer from the command line (all 19 feature flags are required):

    python3 predict.py --single \\
        --gender Female --SeniorCitizen No --Partner Yes --Dependents No --tenure 5 \\
        --PhoneService Yes --MultipleLines No --InternetService "Fiber optic" \\
        --OnlineSecurity No --OnlineBackup No --DeviceProtection No --TechSupport No \\
        --StreamingTV No --StreamingMovies No --Contract Month-to-month \\
        --PaperlessBilling Yes --PaymentMethod "Electronic check" \\
        --MonthlyCharges 85.5 --TotalCharges 427.5

Both modes print a churn-risk probability (0-1) and a Yes/No prediction at the default 0.5
threshold. See Section 1.3 of the notebook for the cost-asymmetry reasoning behind trying a lower
threshold (e.g. 0.35) in a real retention workflow, where missing a churner is costlier than a
false alarm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model.joblib"

# The raw feature columns the pipeline's ColumnTransformer expects, in the order used during
# training. Declared explicitly (rather than inferred at runtime) so a malformed --single call
# fails fast with a clear error instead of a confusing sklearn stack trace.
FEATURE_COLUMNS: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

# The six add-on service columns summed into the engineered `services_count` feature
# (notebook Section 3.3) — declared once here so the CLI's raw input and the engineered feature
# always agree on which columns count as "a service".
SERVICE_COLUMNS: list[str] = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns the fitted pipeline's ColumnTransformer actually expects, including the two engineered
# features added on top of FEATURE_COLUMNS in Section 3.3. Kept separate from FEATURE_COLUMNS so
# --single's required-flag list (FEATURE_COLUMNS) still reflects only what a human must type in —
# the engineered columns are always derived, never asked for.
MODEL_INPUT_COLUMNS: list[str] = FEATURE_COLUMNS + ["services_count", "is_new_customer"]


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained sklearn Pipeline (preprocessing + classifier) from disk.

    Raises FileNotFoundError with a clear message rather than letting joblib's generic error
    surface, since the most common failure mode is running this script from the wrong directory.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Run this script from the `report/` directory, "
            f"or pass --model with an explicit path to model.joblib."
        )
    return joblib.load(model_path)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning the model was trained on (notebook Section 3) to raw input rows.

    The saved pipeline's OneHotEncoder was fit on `SeniorCitizen` as "Yes"/"No" strings and on
    `TotalCharges` as a numeric column with new-customer blanks filled as 0.0 — raw exports from
    the source system (e.g. `SeniorCitizen` as 0/1, or `TotalCharges` as a string with blanks for
    brand-new customers) must be normalised to match, or the encoder raises a dtype error on
    unrecognised category values instead of silently mis-scoring.
    """
    cleaned = df.copy()
    # Check for *numeric* dtype explicitly (rather than `!= object`) — pandas can represent
    # string columns as either classic `object` dtype or the newer `StringDtype` depending on
    # version/backend, so "not object" is not a reliable test for "is actually numeric".
    if pd.api.types.is_numeric_dtype(cleaned["SeniorCitizen"]):
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].map({0: "No", 1: "Yes"})
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce").fillna(0.0)
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce notebook Section 3.3's two engineered features on raw (cleaned) input rows.

    `services_count`: how many of the 6 optional add-ons a customer has ("Yes" in that column).
    `is_new_customer`: 1 if the customer is in their first 6 months, else 0 — lets the model use
    the sharp early-tenure risk drop-off (Section 2.4) without relying on tree models alone to
    discover the threshold.
    """
    engineered = df.copy()
    engineered["services_count"] = (engineered[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    engineered["is_new_customer"] = (engineered["tenure"].astype(float) <= 6).astype(int)
    return engineered


def score_dataframe(model: Any, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Vectorised scoring of every row in `df` at once (no per-row Python loop).

    Returns a copy of `df` with two new columns: `churn_probability` and `churn_prediction`.
    """
    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {sorted(missing_cols)}")

    clean_df = clean_features(df[FEATURE_COLUMNS])
    engineered_df = engineer_features(clean_df)
    probabilities = model.predict_proba(engineered_df[MODEL_INPUT_COLUMNS])[:, 1]
    scored = df.copy()
    scored["churn_probability"] = probabilities.round(4)
    scored["churn_prediction"] = np.where(probabilities >= threshold, "Yes", "No")
    return scored


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to model.joblib")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for the Yes/No label (default 0.5; see notebook Section 1.3)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="CSV of raw customer rows to score in bulk")
    mode.add_argument("--single", action="store_true", help="Score one customer from CLI flags below")

    parser.add_argument("--output", type=Path, default=None,
                         help="Where to write the scored CSV (bulk mode only). Defaults to <input>_scored.csv")

    for col in FEATURE_COLUMNS:
        parser.add_argument(f"--{col}", type=str, default=None, help=f"(--single mode) value for '{col}'")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    model = load_model(args.model)

    if args.input is not None:
        raw_df = pd.read_csv(args.input)
        scored_df = score_dataframe(model, raw_df, threshold=args.threshold)
        output_path = args.output or args.input.with_name(f"{args.input.stem}_scored.csv")
        scored_df.to_csv(output_path, index=False)
        n_high_risk = int((scored_df["churn_prediction"] == "Yes").sum())
        print(f"Scored {len(scored_df)} customers -> {output_path}")
        print(f"Flagged as high churn risk (threshold={args.threshold}): {n_high_risk}")
        return

    # --single mode: build a one-row DataFrame from CLI flags, coercing the two numeric
    # columns explicitly since argparse reads every flag as a string.
    missing = [c for c in FEATURE_COLUMNS if getattr(args, c) is None]
    if missing:
        parser.error(f"--single requires all customer feature flags. Missing: {missing}")

    row = {c: getattr(args, c) for c in FEATURE_COLUMNS}
    row["tenure"] = float(row["tenure"])
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    row["TotalCharges"] = float(row["TotalCharges"])
    single_df = pd.DataFrame([row])

    scored_df = score_dataframe(model, single_df, threshold=args.threshold)
    probability = float(scored_df.loc[0, "churn_probability"])
    prediction = scored_df.loc[0, "churn_prediction"]
    print(f"Churn probability: {probability:.2%}")
    print(f"Prediction (threshold={args.threshold}): {prediction}")


if __name__ == "__main__":
    main()
