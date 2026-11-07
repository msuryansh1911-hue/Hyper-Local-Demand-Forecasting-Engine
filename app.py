"""Step 3: Streamlit dashboard. Run:  streamlit run app.py"""
import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import config
from src.features import build_features

st.set_page_config(page_title="Dark Store Demand Forecast", page_icon="📦", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(config.MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner="Fetching live Open-Meteo forecast...")
def fetch_forecast(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon, "forecast_days": 7,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": config.TIMEZONE,
    }
    r = requests.get(config.OPEN_METEO_FORECAST, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": d["time"], "temp_max": d["temperature_2m_max"],
                         "temp_min": d["temperature_2m_min"],
                         "precip_mm": [p or 0.0 for p in d["precipitation_sum"]]})


def recommendations(fc: pd.DataFrame, pred: pd.DataFrame, heat_c: float, rain_mm: float, hol_names: dict):
    recs = []
    day = lambda s: pd.to_datetime(s).strftime("%a %d %b")
    hot = fc[fc["temp_max"] >= heat_c]
    wet = fc[fc["precip_mm"] >= rain_mm]
    cool = fc[fc["temp_max"] < 26]
    if not hot.empty:
        recs.append(("🔥", f"Stockpile **Cold Beverages** and **Ice Cream**: up to "
                           f"{hot['temp_max'].max():.0f}°C expected on {', '.join(day(d) for d in hot['date'])}. "
                           "Pre-chill extra stock and push cooling bundles in-app."))
    if not wet.empty:
        recs.append(("🌧️", f"Rain expected on {', '.join(day(d) for d in wet['date'])} "
                           f"(peak {wet['precip_mm'].max():.0f} mm). Stock **Umbrellas & Rainwear** near the "
                           "dispatch bay, add **Hot Beverages & Instant Noodles** to the home carousel, and "
                           "pre-position extra riders (delivery times degrade in rain)."))
    if not cool.empty and wet.empty:
        recs.append(("🧥", f"Cooler days ({', '.join(day(d) for d in cool['date'])}): lift hot beverage stock."))
    for d, name in hol_names.items():
        recs.append(("🎉", f"**{name}** on {day(d)}: expect snack/party-category lift; confirm supplier "
                           "delivery the day before."))
    peak = pred.groupby("date")["forecast"].sum().idxmax()
    recs.append(("📈", f"Overall peak demand day: **{day(peak)}** "
                       f"({pred[pred['date'] == peak]['forecast'].sum():,.0f} units). Schedule max staffing."))
    return recs


st.title("📦 Hyper-Local Demand Forecasting Engine")
st.caption("Quick-commerce dark store · 7-day category demand · live Open-Meteo weather forecast")

if not config.MODEL_PATH.exists():
    st.error("Model not found. Run `python src/ingest.py` then `python src/train.py` first.")
    st.stop()
bundle = load_bundle()

with st.sidebar:
    st.header("Store Manager Controls")
    city = st.selectbox("Dark store location", list(config.CITIES), index=0)
    heat_c = st.slider("Heatwave threshold (°C)", 28, 42, int(config.HEATWAVE_C))
    rain_mm = st.slider("Rain threshold (mm/day)", 0.5, 20.0, float(config.RAIN_MM), 0.5)
    safety = st.slider("Safety stock buffer (%)", 0, 50, 15)
    st.subheader("What-if scenario")
    dtemp = st.slider("Temperature shift (°C)", -5, 8, 0)
    extra_rain = st.slider("Add rain on every day (mm)", 0, 30, 0)
    cats = st.multiselect("Categories", list(bundle["models"]), default=list(bundle["models"]))

try:
    lat, lon = config.CITIES[city]
    fc = fetch_forecast(lat, lon).copy()
except Exception as e:
    st.error(f"Could not reach Open-Meteo: {e}")
    st.stop()

fc["temp_max"] += dtemp
fc["temp_min"] += dtemp
fc["precip_mm"] += extra_rain
fc["is_holiday"] = fc["date"].isin(bundle["holiday_dates"]).astype(int)

feat = build_features(fc, bundle["holiday_dates"])
feat["is_raining"] = (feat["precip_mm"] >= rain_mm).astype(int)   # user-adjustable thresholds
feat["is_heatwave"] = (feat["temp_max"] >= heat_c).astype(int)

rows = []
for cat in cats:
    p = bundle["models"][cat].predict(feat[bundle["feature_cols"]]).clip(min=0)
    rows.append(pd.DataFrame({"date": feat["date"], "category": cat, "forecast": p.round(0)}))
if not rows:
    st.warning("Select at least one category.")
    st.stop()
pred = pd.concat(rows, ignore_index=True)
pred["stock_target"] = (pred["forecast"] * (1 + safety / 100)).round(0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("7-day forecast units", f"{pred['forecast'].sum():,.0f}")
c2.metric("Stock target (+buffer)", f"{pred['stock_target'].sum():,.0f}")
c3.metric("Hottest day", f"{feat['temp_max'].max():.0f}°C")
c4.metric("Total rain", f"{feat['precip_mm'].sum():.0f} mm")

left, right = st.columns([3, 2])
with left:
    st.subheader("Forecast by category")
    fig = px.line(pred, x="date", y="forecast", color="category", markers=True)
    fig.update_layout(legend_title_text="", margin=dict(t=10), yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Weather driving the forecast")
    w = feat.melt(id_vars="date", value_vars=["temp_max", "precip_mm"])
    fig2 = px.bar(w[w["variable"] == "precip_mm"], x="date", y="value", labels={"value": "Rain (mm)"})
    fig2.add_scatter(x=feat["date"], y=feat["temp_max"], name="Max °C", yaxis="y2", mode="lines+markers")
    fig2.update_layout(yaxis2=dict(overlaying="y", side="right", title="°C"), margin=dict(t=10),
                       showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("💡 Growth & inventory recommendations")
hol_days = feat[feat["is_holiday"] == 1]["date"].tolist()
hol_names = {d: bundle["holiday_names"].get(d, "Public holiday") for d in hol_days}
for icon, text in recommendations(feat, pred, heat_c, rain_mm, hol_names):
    st.info(text, icon=icon)

st.subheader("Stock plan")
plan = pred.pivot(index="category", columns="date", values="stock_target")
plan["7-day total"] = plan.sum(axis=1)
st.dataframe(plan.astype(int), use_container_width=True)
st.download_button("Download stock plan (CSV)", plan.to_csv().encode(), "stock_plan.csv")

with st.expander("Model quality (holdout: last 28 days)"):
    st.dataframe(pd.DataFrame(bundle["metrics"]).T.rename(columns={
        "mae": "MAE", "mape_pct": "MAPE %", "baseline_mae": "Day-of-week baseline MAE"}))
    st.caption(f"Trained through {bundle['trained_through']}. Training sales are synthetic; weather and "
               "holidays are real (Open-Meteo, Nager.Date).")
