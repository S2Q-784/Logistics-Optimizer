import pandas as pd
import numpy as np
from pathlib import Path

from eda_utils import clean_cols, try_parse_dates, add_time_parts, map_priority, safe_div, detect_target


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT  = DATA / "model_ready_dataset.csv"

def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name)
    df = clean_cols(df)
    df = try_parse_dates(df)
    return df

def main():
    # ---- Load all seven source tables
    orders = load("orders.csv")
    deliv  = load("delivery_performance.csv")
    routes = load("routes_distance.csv")
    costs  = load("cost_breakdown.csv")
    fb     = load("customer_feedback.csv")
    veh    = load("vehicle_fleet.csv")
    wh     = load("warehouse_inventory.csv")   # currently not joined (kept for future FE)

    # ---- Build the order-centric fact table
    df = (
        orders
        .merge(deliv,  on="order_id", how="left")
        .merge(routes, on="order_id", how="left")
        .merge(costs,  on="order_id", how="left")
        .merge(fb,     on="order_id", how="left")
    )

    # Optional vehicle enrichment (if vehicle_id present on both)
    if "vehicle_id" in df.columns and "vehicle_id" in veh.columns:
        df = df.merge(veh, on="vehicle_id", how="left")

    # ---- Target: is_delayed (robust rules in eda_utils.detect_target)
    df["is_delayed"] = detect_target(df)

    # ---- Light feature engineering used downstream
    # priority encoding
    if "priority" in df.columns:
        df["priority_num"] = df["priority"].apply(map_priority)

    # lead time in days when both columns exist
    if "promised_delivery_date" in df.columns and "order_date" in df.columns:
        df["lead_time_days"] = (
            pd.to_datetime(df["promised_delivery_date"]) -
            pd.to_datetime(df["order_date"])
        ).dt.days

    # cost/distance ratios
    if {"total_cost","distance_km"}.issubset(df.columns):
        df["cost_per_km"] = safe_div(df["total_cost"], df["distance_km"])
    if {"fuel_cost","distance_km"}.issubset(df.columns):
        df["fuel_cost_per_km"] = safe_div(df["fuel_cost"], df["distance_km"])

    # add time parts for common date columns (used in EDA/modeling)
    for c in ["order_date", "dispatch_date", "promised_delivery_date", "scheduled_delivery_date"]:
        if c in df.columns:
            df = add_time_parts(df, c)

    # ---- Save the final dataset
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT} with shape {df.shape}")

if __name__ == "__main__":
    main()
