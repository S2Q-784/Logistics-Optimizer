import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
IN_CSV = DATA / "model_ready_dataset.csv"
FEAT_OUT = DATA / "feature_lists.json"
EVAL_OUT = DATA / "train_test_eval.json"
CONF_TRAIN = DATA / "confusion_train.csv"
CONF_TEST = DATA / "confusion_test.csv"

def drop_leakage_and_select(df: pd.DataFrame):
    df = df.copy()
    if "is_delayed" not in df.columns:
        raise ValueError("Target `is_delayed` not found.")
    y = (df["is_delayed"] > 0).astype(int)

    # drop target + raw datetimes (use engineered parts instead)
    drop = {"is_delayed"} | {c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)}

    # leakage patterns & IDs
    leak_patterns = ["actual","delivered","delay","late","arrival","arrived","completed","status"]
    for c in df.columns:
        lc = c.lower()
        if any(p in lc for p in leak_patterns): drop.add(c)
        if lc.endswith("_id") or lc.startswith("id_") or lc in {"id","order_id","vehicle_id"}:
            drop.add(c)

    X = df.drop(columns=list(drop), errors="ignore")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    # remove high-cardinality free text (heuristic)
    high_text = [c for c in cat_cols if X[c].nunique(dropna=True) > 0.5*len(X)]
    if high_text:
        X = X.drop(columns=high_text, errors="ignore")
        cat_cols = [c for c in cat_cols if c not in high_text]

    return X, y, num_cols, cat_cols, sorted(list(drop)), high_text

def metrics_all(y_true, y_pred, y_score):
    return {
        "accuracy": float(accuracy_score(y_true,y_pred)),
        "precision": float(precision_score(y_true,y_pred, zero_division=0)),
        "recall": float(recall_score(y_true,y_pred,   zero_division=0)),
        "f1": float(f1_score(y_true,y_pred,           zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }

def main():
    df = pd.read_csv(IN_CSV)
    X, y, num_cols, cat_cols, dropped_cols, dropped_text = drop_leakage_and_select(df)

    # Save feature lists (so you can show them)
    feature_lists = {
        "input_features_numeric": num_cols,
        "input_features_categorical": cat_cols,
        "dropped_leakage_or_ids": dropped_cols,
        "dropped_high_card_text": dropped_text,
        "output_feature": "is_delayed"
    }
    Path(FEAT_OUT).write_text(json.dumps(feature_lists, indent=2))

    # Split
    Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # --- Imputation + Scaling + Encoding ---
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=False))  # keep sparse compatibility
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

    # --- Regularization + Early Stopping ---
    clf = SGDClassifier(
        loss="log_loss",          # logistic regression
        penalty="elasticnet",     # L1 + L2
        alpha=0.0005,             # regularization strength
        l1_ratio=0.15,            # L1/L2 mix
        early_stopping=True,      # <- early stopping on an internal val split
        validation_fraction=0.2,
        n_iter_no_change=5,
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, Ytr)

    # Train eval
    y_pred_tr = pipe.predict(Xtr)
    y_score_tr = pipe.decision_function(Xtr)
    m_train = metrics_all(Ytr, y_pred_tr, y_score_tr)
    cm_tr = confusion_matrix(Ytr, y_pred_tr)
    pd.DataFrame(cm_tr, index=["actual_0","actual_1"], columns=["pred_0","pred_1"]).to_csv(CONF_TRAIN)

    # Test eval
    y_pred_te = pipe.predict(Xte)
    y_score_te = pipe.decision_function(Xte)
    m_test = metrics_all(Yte, y_pred_te, y_score_te)
    cm_te = confusion_matrix(Yte, y_pred_te)
    pd.DataFrame(cm_te, index=["actual_0","actual_1"], columns=["pred_0","pred_1"]).to_csv(CONF_TEST)

    Path(EVAL_OUT).write_text(json.dumps({
        "input_features_numeric": num_cols,
        "input_features_categorical": cat_cols,
        "output_feature": "is_delayed",
        "train_metrics": m_train,
        "test_metrics": m_test,
        "confusion_matrix_train_csv": str(CONF_TRAIN),
        "confusion_matrix_test_csv": str(CONF_TEST)
    }, indent=2))

    print(json.dumps({
        "train_metrics": m_train,
        "test_metrics": m_test
    }, indent=2))

if __name__ == "__main__":
    main()
