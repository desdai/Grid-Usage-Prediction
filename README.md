# Grid Usage Prediction with Spark ML and Weather Augmentation

## Project Overview
This project compares two approaches for short-term electricity load prediction on PJM regional energy data:

1. **Baseline model** using historical load and time-based features only  
2. **Augmented model** using the same baseline features plus external historical weather features  

The goal is to test whether adding external weather information improves hourly grid usage prediction.

---

## Project Idea
Electricity demand is not determined only by historical usage patterns. It is also influenced by external conditions, especially weather. Extreme heat and extreme cold can increase energy demand because of cooling and heating needs.

To study this, we built two Spark ML models on one year of COMED hourly load data:

- a **baseline Spark ML model**
- a **weather-augmented Spark ML model**

We then compared their performance using standard regression metrics.

---

## Dataset
### Load Data
We use the **Hourly Energy Consumption** dataset from Kaggle, which contains real hourly electricity load data from PJM regional zones.

For this project, we use:

- `data/COMED_hourly.csv`

and focus on the date range:

- `2018-01-01` to `2018-08-03`

### External Weather Data
We use historical Chicago weather data as an external signal for the COMED region.

Weather features are automatically retrieved using the `meteostat` Python package and include:

- temperature
- dew point
- relative humidity
- precipitation
- snow depth
- wind speed
- pressure
- extreme heat flag
- extreme cold flag

---

## Methods

### 1. Baseline Model
The baseline Spark ML model uses only historical load and calendar features:

- hour
- day of week
- month
- weekend indicator
- previous 1 hour load
- previous 24 hour load
- previous 168 hour load
- rolling mean over previous 24 hours
- rolling mean over previous 168 hours

### 2. Augmented Model
The augmented Spark ML model uses all baseline features plus weather features:

- temperature
- dew point
- relative humidity
- precipitation
- snow depth
- wind speed
- pressure
- extreme heat flag
- extreme cold flag

### 3. Model
Both models are trained using **Spark MLlib RandomForestRegressor**.

### 4. Evaluation
We compare the two models using:

- MAE
- RMSE
- R²

---

## Project Structure

```text
Grid Usage Prediction/
│
├── data/
│   ├── COMED_hourly.csv
│   └── comed_weather_2018.csv
│
├── outputs_spark_simple/
│   └── comed_2018/
│       ├── metrics.json
│       ├── predictions.csv
│       └── plots/
│
├── build_weather_features.py
├── spark_train_simple.py
├── plot_simple.py
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
pip install meteostat
```

If PySpark is not installed correctly, use:

```bash
pip uninstall -y pyspark
pip install pyspark==3.5.6
```

---

## How to Run

### Step 1. Build weather features
```bash
python .\build_weather_features.py --start_date 2018-01-01 --end_date 2018-08-03 --output_csv ".\data\comed_weather_2018.csv"
```

### Step 2. Train baseline and augmented Spark ML models
```bash
python .\spark_train_simple.py --csv ".\data\COMED_hourly.csv" --weather_csv ".\data\comed_weather_2018.csv" --output_dir ".\outputs_spark_simple\comed_2018" --start_date 2018-01-01 --end_date 2018-08-03
```

### Step 3. Generate plots
```bash
python .\plot_simple.py --predictions_csv ".\outputs_spark_simple\comed_2018\predictions.csv" --output_dir ".\outputs_spark_simple\comed_2018\plots"
```

---

## Results

### Baseline Model
- **MAE:** 779.97
- **RMSE:** 1052.99
- **R²:** 0.8676

### Augmented Model
- **MAE:** 754.82
- **RMSE:** 1008.89
- **R²:** 0.8785

### Summary
The weather-augmented model outperformed the baseline model on all three metrics.

This suggests that adding historical weather information improves hourly load prediction for the COMED region.

---

## Visualization
The project includes plots comparing:

- actual vs predicted load
- baseline vs augmented performance
- zoomed views of selected time periods

These visualizations are saved in:

```text
outputs_spark_simple/comed_2018/plots/
```

---

## Key Finding
The main finding of this project is that **external weather information improves grid usage prediction compared with using historical load alone**.

Even with a simple one-region, one-year setup, the weather-augmented Spark ML model achieved better predictive performance than the baseline model.

---

## Notes
- This project uses **Spark ML** to satisfy the cloud computing course requirement.
- We intentionally simplified the scope to one region and one year to make the pipeline reliable and easy to reproduce.
- The external signal is weather rather than news retrieval because weather is more directly related to load fluctuations and is more reliable to collect automatically.

---

## Future Work
Possible extensions include:

- testing multiple PJM regions
- extending to longer time ranges
- trying additional Spark ML models
- adding more external features such as holidays or severe weather alerts
- building an interactive dashboard for model comparison
