"""Shared configuration for the Hyper-Local Demand Forecasting Engine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "forecast.db"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
SCHEMA_PATH = ROOT / "src" / "schema.sql"

TIMEZONE = "Asia/Kolkata"
COUNTRY_CODE = "IN"
HISTORY_DAYS = 183  # ~6 months

CITIES = {
    "Bengaluru (Koramangala)": (12.9352, 77.6245),
    "Delhi (Connaught Place)": (28.6315, 77.2167),
    "Mumbai (Bandra)": (19.0596, 72.8295),
}
DEFAULT_CITY = "Bengaluru (Koramangala)"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
NAGER_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{cc}"

# category -> baseline daily units for a mid-size dark store
CATEGORIES = {
    "Cold Beverages": 220,
    "Ice Cream": 120,
    "Umbrellas & Rainwear": 25,
    "Hot Beverages & Instant Noodles": 150,
    "Snacks": 300,
    "Grocery Staples": 400,
}

HEATWAVE_C = 33.0
RAIN_MM = 2.0
