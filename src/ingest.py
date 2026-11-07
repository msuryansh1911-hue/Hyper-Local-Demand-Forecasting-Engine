"""Step 1: fetch weather + holidays, synthesize sales, load into SQLite.

Usage:  python src/ingest.py [--city "Bengaluru (Koramangala)"]
"""
import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# Fallback: Nager.Date does not support India (HTTP 204). Major national/festival dates, 2026-27
# (festival dates follow the lunar calendar; verify against an official list for production use).
INDIA_HOLIDAYS = [
    ("2026-01-26", "Republic Day"), ("2026-03-04", "Holi"), ("2026-03-21", "Eid al-Fitr"),
    ("2026-04-03", "Good Friday"), ("2026-08-15", "Independence Day"),
    ("2026-09-04", "Janmashtami"), ("2026-09-14", "Ganesh Chaturthi"),
    ("2026-10-02", "Gandhi Jayanti"), ("2026-10-20", "Dussehra"),
    ("2026-11-01", "Kannada Rajyotsava"), ("2026-11-08", "Diwali"), ("2026-12-25", "Christmas"),
    ("2027-01-26", "Republic Day"), ("2027-03-22", "Holi"), ("2027-08-15", "Independence Day"),
    ("2027-10-02", "Gandhi Jayanti"), ("2027-10-29", "Diwali"), ("2027-12-25", "Christmas"),
]


def fetch_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": config.TIMEZONE,
    }
    r = requests.get(config.OPEN_METEO_ARCHIVE, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": d["time"],
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "precip_mm": d["precipitation_sum"],
    })
    df["precip_mm"] = df["precip_mm"].fillna(0.0)
    return df.dropna(subset=["temp_max", "temp_min"]).reset_index(drop=True)


def fetch_holidays(years) -> pd.DataFrame:
    rows = []
    for y in years:
        try:
            r = requests.get(config.NAGER_URL.format(year=y, cc=config.COUNTRY_CODE), timeout=20)
            r.raise_for_status()
            rows += [{"date": h["date"], "name": h["name"]} for h in r.json()]
        except Exception as e:  # holidays are a nice-to-have; don't kill the pipeline
            print(f"[warn] holidays for {y} unavailable: {e}")
    if not rows:
        print("[info] Nager.Date has no data for this country (it does not cover India); "
              "using curated Indian festival calendar.")
        rows = [{"date": d, "name": n} for d, n in INDIA_HOLIDAYS]
    return pd.DataFrame(rows).drop_duplicates("date")


def synthesize_sales(weather: pd.DataFrame, holidays: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic baseline: units = base * multipliers, with negative-binomial-ish noise.

    NOTE: synthetic by design (no public dark-store data). Effects are domain assumptions.
    """
    rng = np.random.default_rng(seed)
    hol = set(holidays["date"])
    d = pd.to_datetime(weather["date"])
    dow = d.dt.day_of_week.values
    heat = np.clip((weather["temp_max"].values - 28) / 8, 0, 1.5)   # ramps above 28C
    cool = np.clip((26 - weather["temp_max"].values) / 6, 0, 1.0)
    rain = (weather["precip_mm"].values >= config.RAIN_MM).astype(float)
    heavy = np.clip(weather["precip_mm"].values / 20, 0, 1.0)
    weekend = (dow >= 5).astype(float)
    is_hol = weather["date"].isin(hol).values.astype(float)
    trend = np.linspace(0.95, 1.10, len(weather))  # mild growth

    mult = {
        "Cold Beverages":  1 + 0.9 * heat - 0.15 * rain + 0.10 * weekend + 0.15 * is_hol,
        "Ice Cream":       1 + 0.8 * heat - 0.20 * rain + 0.25 * weekend + 0.20 * is_hol,
        "Umbrellas & Rainwear": 1 + 3.0 * rain + 2.0 * heavy,
        "Hot Beverages & Instant Noodles": 1 + 0.5 * cool + 0.6 * rain + 0.4 * heavy,
        "Snacks":          1 + 0.20 * weekend + 0.25 * is_hol + 0.15 * rain,
        "Grocery Staples": 1 + 0.10 * weekend - 0.10 * is_hol + 0.05 * rain,
    }
    out = []
    for cat, base in config.CATEGORIES.items():
        lam = base * trend * mult[cat]
        # gamma-Poisson mixture = negative binomial (overdispersed demand)
        units = rng.poisson(rng.gamma(shape=100, scale=lam / 100))
        out.append(pd.DataFrame({"date": weather["date"], "category": cat, "units_sold": units}))
    return pd.concat(out, ignore_index=True)


def load_db(weather, holidays, sales) -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.DB_PATH) as con:
        con.executescript(config.SCHEMA_PATH.read_text(encoding="utf-8"))
        weather.to_sql("weather_daily", con, if_exists="append", index=False)
        holidays.to_sql("holidays", con, if_exists="append", index=False)
        sales.to_sql("sales_daily", con, if_exists="append", index=False)
        for t in ("weather_daily", "holidays", "sales_daily"):
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=config.DEFAULT_CITY, choices=list(config.CITIES))
    args = ap.parse_args()
    lat, lon = config.CITIES[args.city]

    end = date.today() - timedelta(days=5)  # archive API lags a few days
    start = end - timedelta(days=config.HISTORY_DAYS)
    print(f"Fetching weather for {args.city}: {start} -> {end}")
    weather = fetch_weather(lat, lon, start, end)
    print("Fetching holidays (Nager.Date)")
    holidays = fetch_holidays(range(start.year, end.year + 2))
    print("Synthesizing sales baseline")
    sales = synthesize_sales(weather, holidays)
    print(f"Loading into {config.DB_PATH}")
    load_db(weather, holidays, sales)
    print("Done.")


if __name__ == "__main__":
    main()
