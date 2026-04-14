from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", default="outputs_spark_simple/plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["baseline_abs_error"] = (df["actual"] - df["baseline_pred"]).abs()
    df["augmented_abs_error"] = (df["actual"] - df["augmented_pred"]).abs()

    zoom_df = df.tail(24 * 7)

    # 1. Interactive last 7 days plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=zoom_df["Datetime"], y=zoom_df["actual"], mode="lines", name="Actual"))
    fig1.add_trace(go.Scatter(x=zoom_df["Datetime"], y=zoom_df["baseline_pred"], mode="lines", name="Baseline"))
    fig1.add_trace(go.Scatter(x=zoom_df["Datetime"], y=zoom_df["augmented_pred"], mode="lines", name="Augmented"))

    fig1.update_layout(
        title="Actual vs Predicted Load - Last 7 Days",
        xaxis_title="Datetime",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        template="plotly_white",
    )

    fig1.write_html(out_dir / "last_7_days_interactive.html")

    # 2. Interactive absolute error comparison
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["Datetime"], y=df["baseline_abs_error"], mode="lines", name="Baseline Absolute Error"))
    fig2.add_trace(go.Scatter(x=df["Datetime"], y=df["augmented_abs_error"], mode="lines", name="Augmented Absolute Error"))

    fig2.update_layout(
        title="Absolute Error Comparison",
        xaxis_title="Datetime",
        yaxis_title="Absolute Error",
        hovermode="x unified",
        template="plotly_white",
    )

    fig2.write_html(out_dir / "absolute_error_comparison_interactive.html")

    print(f"Saved interactive plots to: {out_dir}")


if __name__ == "__main__":
    main()