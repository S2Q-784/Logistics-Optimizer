import numpy as np
import pandas as pd

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
          .str.replace(r"_+", "_", regex=True)
          .str.strip("_")
    )
    return df

def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            sample = df[c].dropna().astype(str).head(200)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.8:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    return df

def add_time_parts(df, col, prefix=None):
    df = df.copy()
    if col not in df: 
        return df
    s = pd.to_datetime(df[col], errors="coerce")
    p = f"{prefix}_" if prefix else f"{col}_"
    df[f"{p}year"] = s.dt.year
    df[f"{p}month"] = s.dt.month
    df[f"{p}weekday"] = s.dt.weekday
    df[f"{p}is_weekend"] = s.dt.weekday.isin([5,6]).astype(int)
    return df

def map_priority(val):
    if not isinstance(val, str): 
        return np.nan
    return {"economy":1, "standard":2, "express":3}.get(val.strip().lower(), np.nan)

def safe_div(num, den):
    return np.where(den != 0, num/den, np.nan)

def detect_target(df: pd.DataFrame):
    # 1) promised vs actual delivery timestamps (preferred)
    for p in df.columns:
        if "promis" in p and "deliv" in p:
            for a in df.columns:
                if "actual" in a and "deliv" in a:
                    pa = pd.to_datetime(df[p], errors="coerce")
                    aa = pd.to_datetime(df[a], errors="coerce")
                    return (aa > pa).astype(int)
    # 2) status text fallback
    for c in df.columns:
        if "status" in c.lower():
            s = df[c].astype(str).str.lower()
            return (s.str.contains("delay") | s.str.contains("late")).astype(int)
    # 3) otherwise unknown
    return pd.Series(np.nan, index=df.index)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Compact column profiler used by the Streamlit app."""
    out=[]
    for c in df.columns:
        s=df[c]
        d={"column":c,"dtype":str(s.dtype),"missing_pct":round(s.isna().mean()*100,2),"nunique":int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            d.update({"mean":float(s.mean(skipna=True)),"std":float(s.std(skipna=True)),"min":float(s.min(skipna=True)),"max":float(s.max(skipna=True))})
        out.append(d)
    return pd.DataFrame(out)
