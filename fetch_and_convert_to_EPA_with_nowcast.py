# ============================================================
# File: aqi_fetcher.py
# Author: Anas Saleem
# Institution: FAST NUCES
# ============================================================
"""
Purpose:
--------
Fetch historical air pollution data for Karachi using the OpenWeather API
and compute a U.S. EPA–style Air Quality Index (AQI) using official pollutant
breakpoints and averaging periods.

Data Sources:
-------------
1️⃣ OpenWeather Air Pollution API:
    https://openweathermap.org/api/air-pollution  
    - Provides pollutants in µg/m³.
    - Pollutants: pm2_5, pm10, no2, so2, co, o3.

2️⃣ AQI Computation Reference:
    https://openweathermap.org/air-pollution-index-levels  
    - These categories match U.S. EPA breakpoints and health scales.

EPA Considerations:
-------------------
The U.S. EPA defines AQI using pollutant-specific averaging periods and units:
  - PM2.5, PM10  → µg/m³ (NowCast for near real-time data)
  - CO           → ppm (8-hour average)
  - O₃           → ppb (8-hour average; 1-hour for higher levels)
  - SO₂, NO₂     → ppb (1-hour average)

OpenWeather returns all pollutants in µg/m³, so we convert gases to ppb or ppm:
    ppb = (µg/m³ × 24.45) / molecular_weight
    ppm = ppb / 1000  (for CO)

NowCast Explanation:
--------------------
EPA’s NowCast replaces the older 24-hour average for PM2.5 and PM10 when using
hourly data. It weights the most recent hours more heavily, adjusting for
stability in readings. This makes AQI more responsive to recent pollution spikes.

Formula (simplified):
---------------------
weight factor w = min(ratio^11, 0.5), where ratio = (min/ max)
NowCast = sum(concentration_i * w^i) / sum(w^i)

References:
-----------
- U.S. EPA Technical Assistance Document for AQI Reporting (2018)
- EPA NowCast PM Method: https://forum.airnowtech.org/t/the-nowcast-for-pm2-5-and-pm10/172
- OpenWeather Air Quality API Specification
"""

# ============================================================
# Imports
# ============================================================
import os
import math
import requests
import pandas as pd
from datetime import datetime, timezone
import zoneinfo
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# ============================================================
# Environment & Configuration
# ============================================================
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in environment variables!")

# Karachi coordinates
LAT, LON = 24.8546842, 67.0207055

# ============================================================
# EPA AQI Breakpoints
# ============================================================
AQI_BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
    ],
    "pm10": [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300),
        (425, 504, 301, 400), (505, 604, 401, 500)
    ],
    "no2": [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300),
        (1250, 1649, 301, 400), (1650, 2049, 401, 500)
    ],
    "so2": [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300),
        (605, 804, 301, 400), (805, 1004, 401, 500)
    ],
    "co": [
        (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)
    ],
    "o3_8hr": [
        (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
        (86, 105, 151, 200), (106, 200, 201, 300)
    ],
    "o3_1hr": [
        (125, 164, 101, 150), (165, 204, 151, 200),
        (205, 404, 201, 300), (405, 504, 301, 400),
        (505, 604, 401, 500)
    ],
}

# ============================================================
# µg/m³ → ppb/ppm Conversion
# ============================================================
def convert_units(pollutant: str, value: float) -> float:
    """Convert µg/m³ → ppb or ppm based on pollutant."""
    if value is None or isinstance(value, str):
        return None
    if math.isnan(value):
        return None

    molecular_weights = {"co": 28.01, "no2": 46.0055, "so2": 64.066, "o3": 48.00}
    MW = molecular_weights.get(pollutant)
    if MW is None:
        return value  # PM stays in µg/m³

    ppb = (value * 24.45) / MW
    return ppb / 1000 if pollutant == "co" else ppb

# ============================================================
# NowCast Computation (for PM2.5 and PM10)
# ============================================================
def compute_nowcast(series: pd.Series) -> pd.Series:
    """
    Compute NowCast using EPA's algorithm.
    Works on hourly µg/m³ data for PM2.5 and PM10.

    Steps:
    - Consider last 12 hours.
    - ratio = min(conc) / max(conc)
    - weight factor = max(min(ratio**11, 0.5), 0.5)
    - Weighted average = Σ(value_i * w^i) / Σ(w^i)
    """
    nowcast_vals = []
    for i in range(len(series)):
        window = series[max(0, i - 11): i + 1].dropna()
        if len(window) < 3:
            nowcast_vals.append(None)
            continue
        c_min, c_max = window.min(), window.max()
        ratio = c_min / c_max if c_max > 0 else 0
        weight_factor = ratio ** 11 if ratio > 0.5 else 0.5
        weights = [weight_factor ** (len(window) - j - 1) for j in range(len(window))]
        nowcast = (window.values * weights).sum() / sum(weights)
        nowcast_vals.append(nowcast)
    return pd.Series(nowcast_vals, index=series.index)

# ============================================================
# AQI Computation for One Pollutant
# ============================================================
def calc_aqi_for_pollutant(pollutant: str, conc: float) -> int | None:
    """Apply EPA linear interpolation formula."""
    if conc is None or math.isnan(conc):
        return None
    if pollutant not in AQI_BREAKPOINTS:
        return None

    for Cl, Ch, Il, Ih in AQI_BREAKPOINTS[pollutant]:
        if Cl <= conc <= Ch:
            return round(((Ih - Il) / (Ch - Cl)) * (conc - Cl) + Il)

    return 500 if conc > AQI_BREAKPOINTS[pollutant][-1][1] else 0

# ============================================================
# Overall AQI (Composite Max Rule)
# ============================================================
def calc_overall_aqi(row):
    """Compute pollutant-specific AQIs and return the maximum (EPA rule)."""
    aqi_vals = []

    # PMs use NowCast values (replacing 24-hour avg)
    if not pd.isna(row.get("pm2_5_nowcast")):
        aqi_vals.append(calc_aqi_for_pollutant("pm2_5", row["pm2_5_nowcast"]))
    if not pd.isna(row.get("pm10_nowcast")):
        aqi_vals.append(calc_aqi_for_pollutant("pm10", row["pm10_nowcast"]))

    # Gases
    if not pd.isna(row.get("co_ppm_8hr_avg")):
        aqi_vals.append(calc_aqi_for_pollutant("co", row["co_ppm_8hr_avg"]))
    if not pd.isna(row.get("o3_ppb_8hr_avg")):
        o3_val = row["o3_ppb_8hr_avg"]
        if o3_val <= 125:
            aqi_vals.append(calc_aqi_for_pollutant("o3_8hr", o3_val))
        else:
            aqi_vals.append(calc_aqi_for_pollutant("o3_1hr", o3_val))
    if not pd.isna(row.get("so2_ppb")):
        aqi_vals.append(calc_aqi_for_pollutant("so2", row["so2_ppb"]))
    if not pd.isna(row.get("no2_ppb")):
        aqi_vals.append(calc_aqi_for_pollutant("no2", row["no2_ppb"]))

    valid = [v for v in aqi_vals if v is not None]
    return max(valid) if valid else None

# ============================================================
# Fetch Data from OpenWeather
# ============================================================
def fetch_aqi_data(lat: float, lon: float, start_date: datetime, end_date: datetime):
    BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat, "lon": lon,
        "start": int(start_date.timestamp()), "end": int(end_date.timestamp()),
        "appid": API_KEY,
    }

    print(f"\n📡 Fetching AQI data for {lat:.4f}, {lon:.4f}")
    r = requests.get(BASE_URL, params=params)
    if r.status_code != 200:
        print(f"❌ API Error {r.status_code}: {r.text}")
        return None
    print("✅ Data fetched successfully.")
    return r.json()

# ============================================================
# JSON → DataFrame Conversion
# ============================================================
def json_to_dataframe(json_data):
    """Convert OpenWeather JSON → DataFrame with dual unit columns and averaging logic."""
    if not json_data or "list" not in json_data:
        print("⚠️ No data found in API response.")
        return None

    PKT = zoneinfo.ZoneInfo("Asia/Karachi")
    records = []
    for item in json_data["list"]:
        dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc).astimezone(PKT)
        rec = {"datetime": dt, "aqi_owm": item["main"]["aqi"], **item["components"]}
        records.append(rec)

    df = pd.DataFrame(records).sort_values("datetime")

    # --- Converted columns ---
    for gas in ["co", "no2", "so2", "o3"]:
        converted = df[gas].apply(lambda v: convert_units(gas, v))
        unit = "ppm" if gas == "co" else "ppb"
        df[f"{gas}_{unit}"] = converted

    # --- NowCast for PM ---
    df["pm2_5_nowcast"] = compute_nowcast(df["pm2_5"])
    df["pm10_nowcast"] = compute_nowcast(df["pm10"])

    # --- Rolling for other pollutants ---
    df["co_ppm_8hr_avg"] = df["co_ppm"].rolling(8, min_periods=8).mean()
    df["o3_ppb_8hr_avg"] = df["o3_ppb"].rolling(8, min_periods=8).mean()

    # --- AQI computation ---
    df["aqi_epa_calc"] = df.apply(calc_overall_aqi, axis=1)
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    # --- Column order ---
    cols = [
        "datetime", "aqi_owm", "aqi_epa_calc",
        "pm2_5_nowcast",
        "pm10_nowcast",
        "co_ppm", "co_ppm_8hr_avg",
        "o3_ppb", "o3_ppb_8hr_avg",
        "no2_ppb", "so2_ppb"
    ]
    return df[cols]

# ============================================================
# Rich Console Display
# ============================================================
def display_table_rich(df, last_n=None):
    console = Console(force_terminal=True, width=200)
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    for col in df.columns:
        table.add_column(col, justify="right" if pd.api.types.is_numeric_dtype(df[col]) else "left")
    subset = df.tail(last_n) if last_n else df
    for _, row in subset.iterrows():
        values = ["-" if pd.isna(v) else f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        table.add_row(*values)
    console.print(table)

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    end_date = datetime.now(timezone.utc)
    start_date = end_date - relativedelta(months=1)

    data = fetch_aqi_data(LAT, LON, start_date, end_date)
    df = json_to_dataframe(data)
    if df is not None and not df.empty:
        df.dropna(inplace=True)
        display_table_rich(df)
        df.to_csv("karachi_aqi_data_nowcast.csv", index=False)
        print(f"\n💾 Saved {len(df)} records → karachi_aqi_data_nowcast.csv")
        print("ℹ️ Note: First few rows may have NaN until 8/12 hours of data are available.")
    else:
        print("⚠️ No data available for the specified range.")
