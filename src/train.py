"""Step 2: train one gradient-boosting model per category and save with joblib.

Usage:  python src/train.py
"""
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.features import FEATURE_COLS, build_features  # noqa: E402

HOLDOUT_DAYS = 28


def main():
    if not config.DB_PATH.exists():
        sys.exit("Database not found. Run: python src/ingest.py")
    with sqlite3.connect(config.DB_PATH) as con:
        df = pd.read_sql("SELECT * FROM v_training", con)
        hol_df = pd.read_sql("SELECT date, name FROM holidays", con)
    hol = hol_df["date"].tolist()

    df = build_features(df, hol)
    cutoff = (pd.to_datetime(df["date"]).max() - pd.Timedelta(days=HOLDOUT_DAYS)).strftime("%Y-%m-%d")

    models, metrics = {}, {}
    for cat, g in df.groupby("category"):
        train, test = g[g["date"] <= cutoff], g[g["date"] > cutoff]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=5, random_state=42)
        m.fit(train[FEATURE_COLS], train["units_sold"])
        pred = m.predict(test[FEATURE_COLS])
        mae = mean_absolute_error(test["units_sold"], pred)
        mape = float(np.mean(np.abs(test["units_sold"] - pred) / test["units_sold"].clip(lower=1)) * 100)
        # naive baseline: train mean by day-of-week
        dow_mean = train.groupby("dow")["units_sold"].mean()
        base_mae = mean_absolute_error(test["units_sold"], test["dow"].map(dow_mean))
        metrics[cat] = {"mae": round(mae, 2), "mape_pct": round(mape, 2),
                        "baseline_mae": round(base_mae, 2)}
        # refit on all data for production use
        m.fit(g[FEATURE_COLS], g["units_sold"])
        models[cat] = m
        print(f"{cat:34s} MAE={mae:7.2f}  MAPE={mape:5.1f}%  (dow-baseline MAE={base_mae:7.2f})")

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "feature_cols": FEATURE_COLS, "metrics": metrics,
                 "holiday_dates": hol,
                 "holiday_names": dict(zip(hol_df["date"], hol_df["name"])),"trained_through": df["date"].max()}, config.MODEL_PATH)
    print(f"Saved model bundle -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
