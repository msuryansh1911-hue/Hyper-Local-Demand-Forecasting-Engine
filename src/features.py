"""Shared feature engineering (used by train.py and app.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FEATURE_COLS = [
    "dow", "is_weekend", "month", "is_holiday", "days_to_holiday",
    "temp_max", "temp_min", "precip_mm", "is_raining", "is_heatwave",
]


def days_to_next_holiday(dates: pd.Series, holiday_dates, cap: int = 14) -> pd.Series:
    hol = np.sort(pd.to_datetime(list(holiday_dates)).values)
    out = []
    for d in pd.to_datetime(dates).values:
        idx = np.searchsorted(hol, d)
        if idx >= len(hol):
            out.append(cap)
        else:
            out.append(min(int((hol[idx] - d) / np.timedelta64(1, "D")), cap))
    return pd.Series(out, index=dates.index)


def build_features(df: pd.DataFrame, holiday_dates) -> pd.DataFrame:
    """df needs: date, temp_max, temp_min, precip_mm, is_holiday."""
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.day_of_week
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = d.dt.month
    df["days_to_holiday"] = days_to_next_holiday(df["date"], holiday_dates)
    df["is_raining"] = (df["precip_mm"] >= config.RAIN_MM).astype(int)
    df["is_heatwave"] = (df["temp_max"] >= config.HEATWAVE_C).astype(int)
    return df
