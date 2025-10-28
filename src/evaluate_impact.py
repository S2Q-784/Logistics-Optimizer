import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

CSV  = r"C:\material\4th year\project\logistics-optimizer\data\model_ready_dataset.csv"

def to_py(x):
    """Convert numpy types to plain Python for JSON serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
        
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost_per_delay", type=float, default=500.0)
    ap.add_argument("--intervention_cost", type=float, default=50.0)
    ap.add_argument("--preventable_fraction", type=float, default=0.6)
    args = ap.parse_args()

    df = pd.read_csv(CSV)

    # Target and features (very simple, no-leakage heuristic)
    y = (df["is_delayed"] > 0).astype(int)
    X = df.select_dtypes(exclude=["datetime"]).drop(columns=["is_delayed"], errors="ignore")

    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if X[c].dtype == "object"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh",  OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])

    model = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=300))])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    # Metrics
    acc  = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec  = recall_score(yte, pred, zero_division=0)
    f1   = f1_score(yte, pred, zero_division=0)
    auc  = roc_auc_score(yte, prob)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

    # Business impact (simple model)
    baseline_cost = int(yte.sum()) * args.cost_per_delay
    prevented     = tp * args.preventable_fraction
    new_cost      = (int(yte.sum()) - prevented) * args.cost_per_delay + (tp + fp) * args.intervention_cost
    savings       = baseline_cost - new_cost
    uplift        = prevented / len(yte)

    report = {
        "metrics": {
            "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "roc_auc": float(auc),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "business_impact": {
            "baseline_cost": float(baseline_cost),
            "new_cost": float(new_cost),
            "estimated_savings": float(savings),
            "on_time_rate_uplift": float(uplift)
        },
        "assumptions": {
            "cost_per_delay": float(args.cost_per_delay),
            "intervention_cost": float(args.intervention_cost),
            "preventable_fraction": float(args.preventable_fraction)
        }
    }

    # Ensure everything is JSON-safe (native Python types)
    report = json.loads(json.dumps(report, default=lambda o: to_py(o)))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
