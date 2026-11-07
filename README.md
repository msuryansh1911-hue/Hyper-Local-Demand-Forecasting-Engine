# Hyper-Local Demand Forecasting Engine

7-day, category-level demand forecasts for a quick-commerce dark store (Bengaluru by default),
driven by live weather forecasts and festival calendars.

**Data honesty:** weather is real (Open-Meteo, no key). Sales are *synthetic* (no public dark-store
data exists) and generated from documented weather/holiday/weekend effects. Nager.Date does not
cover India, so a curated Indian festival calendar is the fallback (`INDIA_HOLIDAYS` in `src/ingest.py`);
for other countries Nager.Date is used live.

## Run from scratch (Windows PowerShell)
```powershell
cd dark-store-forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python src/ingest.py                # fetch weather + holidays, synthesize sales, build data/forecast.db
python src/train.py                 # train per-category models -> models/demand_model.joblib
streamlit run app.py                # opens http://localhost:8501
```
macOS/Linux: `source .venv/bin/activate` instead of the Activate line.

## Layout
- `config.py` locations, categories, thresholds
- `src/schema.sql` tables `weather_daily`, `holidays`, `sales_daily` + view `v_training`
- `src/ingest.py` Step 1 · `src/features.py` shared features · `src/train.py` Step 2 · `app.py` Step 3

## Ideas to extend
Per-SKU models, real order data swap-in (replace `synthesize_sales`), Prophet comparison,
quantile forecasts for safety stock, scheduled re-ingest.
