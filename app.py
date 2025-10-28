import streamlit as st
import pandas as pd
from pathlib import Path
from src.eda_utils import summarize

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")
DATA = Path("data")

st.title("📦 Predictive Delivery Optimizer — EDA & Metrics")

# ---------------- Sidebar: dataset picker ----------------
files = sorted([p for p in DATA.glob("*.csv")])
label_map = {p.name: p for p in files}
default = "model_ready_dataset.csv" if (DATA / "model_ready_dataset.csv").exists() else (files[0].name if files else None)
sel_name = st.sidebar.selectbox("Choose dataset", options=[p.name for p in files], index=[p.name for p in files].index(default) if default else 0)

if sel_name:
    p = label_map[sel_name]
    df = pd.read_csv(p)
    st.subheader(f"Dataset: `{sel_name}`")
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(50), use_container_width=True)

    # Column summary
    with st.expander("🔎 Column summary"):
        st.dataframe(summarize(df), use_container_width=True)

    # Simple charts
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]

    c1, c2 = st.columns(2)
    with c1:
        if num_cols:
            col = st.selectbox("Histogram (numeric)", num_cols)
            st.bar_chart(df[col].dropna().value_counts().sort_index())
    with c2:
        if cat_cols:
            col = st.selectbox("Top categories (categorical)", cat_cols)
            st.bar_chart(df[col].astype(str).value_counts().head(25))

# ---------------- Metrics viewer ----------------
st.markdown("---")
st.header("📈 Model Metrics")

eval_json = DATA / "train_test_eval.json"
robust_json = DATA / "train_test_eval_robust.json"  # if you used the robust script

if eval_json.exists() or robust_json.exists():
    if robust_json.exists():
        st.caption("Showing robust run (threshold tuning / class weights).")
        report = pd.read_json(robust_json)
        # When saved as dict -> dict, just read via json then display
        import json
        report = json.loads(robust_json.read_text())
        best = report.get("best", report)  # supports both formats
        st.json(best)
    else:
        import json
        st.json(json.loads(eval_json.read_text()))
else:
    st.info("No metrics file found yet. Run `python -m src.train_earlystop` or `python -m src.train_robust` first.")

# ---------------- Quick helpers ----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Quick actions")
if st.sidebar.button("Rebuild merged dataset"):
    import subprocess, sys
    cp = subprocess.run([sys.executable, "-m", "src.build_dataset"], capture_output=True, text=True)
    st.sidebar.code(cp.stdout or cp.stderr)
    st.sidebar.success("Rebuilt data/model_ready_dataset.csv")

if st.sidebar.button("Train (EarlyStop)"):
    import subprocess, sys
    cp = subprocess.run([sys.executable, "-m", "src.train_earlystop"], capture_output=True, text=True)
    st.sidebar.code(cp.stdout or cp.stderr)
    st.sidebar.success("Training complete. See train_test_eval.json")

if st.sidebar.button("Train (Robust)"):
    import subprocess, sys
    cp = subprocess.run([sys.executable, "-m", "src.train_robust"], capture_output=True, text=True)
    st.sidebar.code(cp.stdout or cp.stderr)
    st.sidebar.success("Training complete. See train_test_eval_robust.json")
