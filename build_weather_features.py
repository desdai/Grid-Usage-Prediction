from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # COMED -> Chicago
    chicago = Point(41.8781, -87.6298)

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    weather = Hourly(chicago, start, end)
    df = weather.fetch().reset_index()

    if df.empty:
        raise ValueError("No weather data returned. Try a different date range.")

    # Normalize timestamp column name
    if "time" in df.columns:
        df = df.rename(columns={"time": "Datetime"})
    elif "date" in df.columns:
        df = df.rename(columns={"date": "Datetime"})

    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)

    # Keep only the fields we need if they exist
    keep_cols = ["Datetime"]
    for col in ["temp", "dwpt", "rhum", "prcp", "snow", "wspd", "pres"]:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].copy()

    # Rename for clarity
    rename_map = {
        "temp": "temperature",
        "dwpt": "dew_point",
        "rhum": "relative_humidity",
        "prcp": "precipitation",
        "snow": "snow_depth",
        "wspd": "wind_speed",
        "pres": "pressure",
    }
    df = df.rename(columns=rename_map)

    # Simple external indicators
    if "temperature" in df.columns:
        df["is_extreme_heat"] = (df["temperature"] >= 30).astype(int)
        df["is_extreme_cold"] = (df["temperature"] <= -5).astype(int)
    else:
        df["is_extreme_heat"] = 0
        df["is_extreme_cold"] = 0

    if "precipitation" not in df.columns:
        df["precipitation"] = 0.0
    if "snow_depth" not in df.columns:
        df["snow_depth"] = 0.0
    if "wind_speed" not in df.columns:
        df["wind_speed"] = 0.0
    if "relative_humidity" not in df.columns:
        df["relative_humidity"] = 0.0
    if "dew_point" not in df.columns:
        df["dew_point"] = 0.0
    if "pressure" not in df.columns:
        df["pressure"] = 0.0

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved weather features to: {out_path}")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()