from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def get_spark(local_dir: str) -> SparkSession:
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder
        .appName("GridUsageWeather")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "4")
        .config("spark.local.dir", local_dir)
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def load_region_csv(spark: SparkSession, csv_path: str):
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
    )

    if "Datetime" not in df.columns:
        raise ValueError("Input CSV must contain a 'Datetime' column.")

    value_cols = [c for c in df.columns if c != "Datetime"]
    if len(value_cols) != 1:
        raise ValueError(
            f"Expected exactly one load column besides Datetime, found: {value_cols}"
        )

    target_col = value_cols[0]

    df = (
        df.withColumn("Datetime", F.to_timestamp("Datetime"))
        .withColumn(target_col, F.col(target_col).cast("double"))
        .dropna(subset=["Datetime", target_col])
        .dropDuplicates(["Datetime"])
        .orderBy("Datetime")
    )

    return df, target_col


def add_base_features(df, target_col: str):
    df = (
        df.withColumn("date", F.to_date("Datetime"))
        .withColumn("hour", F.hour("Datetime"))
        .withColumn("dayofweek", F.dayofweek("Datetime"))
        .withColumn("month", F.month("Datetime"))
        .withColumn("is_weekend", F.when(F.dayofweek("Datetime").isin([1, 7]), 1).otherwise(0))
    )

    # Single global time series, so no partition key is needed conceptually.
    # This warning is expected for one-series rolling windows.
    w = Window.orderBy("Datetime")
    roll24 = w.rowsBetween(-24, -1)
    roll168 = w.rowsBetween(-168, -1)

    df = (
        df.withColumn("lag_1", F.lag(F.col(target_col), 1).over(w))
        .withColumn("lag_24", F.lag(F.col(target_col), 24).over(w))
        .withColumn("lag_168", F.lag(F.col(target_col), 168).over(w))
        .withColumn("roll_mean_24", F.avg(F.col(target_col)).over(roll24))
        .withColumn("roll_mean_168", F.avg(F.col(target_col)).over(roll168))
        .withColumn("label", F.lead(F.col(target_col), 1).over(w))
    )

    return df


def add_weather_features(df, weather_csv: str | None):
    weather_cols = [
        "temperature",
        "dew_point",
        "relative_humidity",
        "precipitation",
        "snow_depth",
        "wind_speed",
        "pressure",
        "is_extreme_heat",
        "is_extreme_cold",
    ]

    if not weather_csv:
        for c in weather_cols:
            df = df.withColumn(c, F.lit(0.0))
        return df

    spark = df.sparkSession
    weather = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(weather_csv)
        .withColumn("Datetime", F.to_timestamp("Datetime"))
    )

    for c in weather_cols:
        if c not in weather.columns:
            weather = weather.withColumn(c, F.lit(0.0))
        else:
            weather = weather.withColumn(c, F.col(c).cast("double"))

    weather = weather.select("Datetime", *weather_cols)

    df = df.join(weather, on="Datetime", how="left")

    for c in weather_cols:
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

    return df


def chronological_split(df, test_ratio: float = 0.2):
    # Avoid separate count() action. Use percent_rank instead.
    row_w = Window.orderBy("Datetime")
    df = df.withColumn("pct_rank", F.percent_rank().over(row_w))

    train_df = df.filter(F.col("pct_rank") < (1 - test_ratio)).drop("pct_rank")
    test_df = df.filter(F.col("pct_rank") >= (1 - test_ratio)).drop("pct_rank")

    return train_df, test_df


def train_and_predict(train_df, test_df, feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    train_vec = assembler.transform(train_df)
    test_vec = assembler.transform(test_df)

    model = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        numTrees=50,
        maxDepth=8,
        seed=42,
    )

    fitted = model.fit(train_vec)
    preds = fitted.transform(test_vec)

    return preds


def evaluate(preds):
    mae_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    rmse_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    r2_eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    return {
        "MAE": float(mae_eval.evaluate(preds)),
        "RMSE": float(rmse_eval.evaluate(preds)),
        "R2": float(r2_eval.evaluate(preds)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to regional load CSV")
    parser.add_argument("--weather_csv", default=None, help="Path to hourly weather CSV")
    parser.add_argument("--output_dir", default="outputs_spark_simple")
    parser.add_argument("--start_date", default="2018-01-01")
    parser.add_argument("--end_date", default="2018-08-03")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_spark_dir = str((Path.cwd() / "spark_tmp").resolve())
    spark = get_spark(local_spark_dir)
    spark.sparkContext.setLogLevel("ERROR")

    df, target_col = load_region_csv(spark, args.csv)

    df = df.filter(
        (F.col("Datetime") >= F.to_timestamp(F.lit(args.start_date + " 00:00:00")))
        & (F.col("Datetime") <= F.to_timestamp(F.lit(args.end_date + " 23:59:59")))
    )

    df = add_base_features(df, target_col)
    df = add_weather_features(df, args.weather_csv)

    base_features = [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_mean_168",
    ]

    aug_features = base_features + [
        "temperature",
        "dew_point",
        "relative_humidity",
        "precipitation",
        "snow_depth",
        "wind_speed",
        "pressure",
        "is_extreme_heat",
        "is_extreme_cold",
    ]

    df = df.dropna().cache()

    base_df = df.select("Datetime", "date", "label", *base_features)
    aug_df = df.select("Datetime", "date", "label", *aug_features)

    base_train, base_test = chronological_split(base_df, args.test_ratio)
    aug_train, aug_test = chronological_split(aug_df, args.test_ratio)

    print("Training baseline model...")
    base_preds = train_and_predict(base_train, base_test, base_features)

    print("Training augmented model...")
    aug_preds = train_and_predict(aug_train, aug_test, aug_features)

    baseline_metrics = evaluate(base_preds)
    augmented_metrics = evaluate(aug_preds)

    print("\n===== BASELINE =====")
    print(baseline_metrics)

    print("\n===== AUGMENTED =====")
    print(augmented_metrics)

    results = {
        "baseline": baseline_metrics,
        "augmented": augmented_metrics,
        "base_features": base_features,
        "aug_features": aug_features,
        "date_range": {
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    base_pd = base_preds.select(
        "Datetime",
        F.col("label").alias("actual"),
        F.col("prediction").alias("baseline_pred"),
    ).toPandas()

    aug_pd = aug_preds.select(
        "Datetime",
        F.col("prediction").alias("augmented_pred"),
    ).toPandas()

    pred_df = base_pd.merge(aug_pd, on="Datetime", how="inner")
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\nSaved outputs to: {out_dir}")
    spark.stop()


if __name__ == "__main__":
    main()