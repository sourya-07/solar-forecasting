# ☀️ Solar Energy Forecasting

A **Streamlit-based machine learning dashboard** that predicts solar power generation using historical weather and sensor data. The app lets you explore your dataset, visualise generation trends, and train forecasting models — all in your browser.

---

## 🚀 How It Works

```
CSV Data → Clean & Feature Engineer → Train ML Model → Evaluate & Visualise
```

1. **Load** — Upload your own CSV or use the bundled `solar_data.csv`
2. **Preprocess** — Remove duplicates/nulls, engineer time features (hour, season)
3. **Explore** — Browse raw data, statistics, and a correlation heatmap
4. **Visualise** — Analyse power trends, hourly patterns, and seasonal behaviour
5. **Forecast** — Train a model, view accuracy metrics, and compare actual vs. predicted output

---

## 🗂️ Project Structure

```
solar-forecasting/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── data/
│   └── solar_data.csv      # Default dataset (~500 KB)
├── notebooks/
│   └── eda.ipynb           # Exploratory Data Analysis notebook
└── src/
    ├── __init__.py
    ├── data_loader.py      # CSV loading utilities
    ├── preprocessing.py    # Cleaning, feature engineering, train/test split
    ├── model.py            # Model training & evaluation
    └── visualization.py   # Plotly chart builders
```

---

## 📦 Installation

> **Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd solar-forecasting

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## 🖥️ Dashboard Tabs

### 1. 📊 Data Explorer

| Section | Description |
|---|---|
| **Metrics** | Quick summary — total rows, columns, and average power output |
| **Raw Data Preview** | First 100 rows of the cleaned dataset |
| **Summary Statistics** | Mean, std, min/max for every feature |
| **Correlation Heatmap** | Pearson correlations across all numeric features |

### 2. 📈 Trends & Seasonality

| Chart | Description |
|---|---|
| **Power Generation Trend** | Line plot of `generated_power_kw` over time |
| **Hourly Pattern** | Box plot grouped by hour of day (10 readings/day) |
| **Seasonal Pattern** | Box plot grouped by season (Winter / Spring / Summer / Autumn) |

### 3. 🤖 Forecasting

1. **Choose a model** — *Linear Regression* or *Random Forest*
2. **Set test size** — Slider from 10 % to 40 % (default 20 %)
3. **Click "🚀 Train Model"** — Model trains on the chronological train split and reports:

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error in kW |
| **RMSE** | Root Mean Squared Error in kW |
| **R² Score** | Coefficient of determination (1.0 = perfect fit) |

4. **Charts produced:**
   - *Actual vs Predicted* — line overlay of true and forecast values
   - *Scatter plot* — predicted vs. actual with a perfect-fit reference line
   - *Feature Importance* (Random Forest only) — ranked bar chart of input feature weights

---

## 🛠️ Source Modules

### `src/data_loader.py`
```python
load_csv(filepath)          # Load a CSV from disk
load_uploaded_file(file)    # Load a Streamlit UploadedFile object
```

### `src/preprocessing.py`
| Function | Purpose |
|---|---|
| `clean_data(df)` | Drop duplicates & NaN rows |
| `add_time_features(df)` | Add `hour_of_day`, `day_index`, and `season` columns |
| `get_feature_columns(df)` | Return the list of usable feature columns |
| `prepare_data(df, test_size)` | Full pipeline → scaled X/y train-test arrays |

**Feature set (20 raw + 3 engineered):**
- Weather: temperature, humidity, pressure, precipitation, snowfall
- Cloud cover: surface, high, medium, low layers
- Radiation: shortwave backwards
- Wind: speed & direction at 10 m, 80 m, and 900 mb; gust at 10 m
- Solar geometry: angle of incidence, zenith, azimuth
- Engineered: `hour_of_day`, `day_index`, `season`

### `src/model.py`
```python
train_linear_regression(X_train, y_train)   # sklearn LinearRegression
train_random_forest(X_train, y_train)        # RandomForestRegressor (n_jobs=-1)
predict(model, X)                            # Generate predictions
evaluate_model(model, X_test, y_test)        # Returns MAE, RMSE, R²
```

### `src/visualization.py`
| Function | Chart Type |
|---|---|
| `plot_power_trend(df)` | Line chart — power over time |
| `plot_hourly_pattern(df)` | Box plot — power by hour |
| `plot_seasonal_pattern(df)` | Box plot — power by season |
| `plot_correlation_heatmap(df)` | Heatmap — feature correlations |
| `plot_actual_vs_predicted(y_true, y_pred)` | Dual-line chart |
| `plot_scatter_actual_vs_predicted(y_true, y_pred)` | Scatter + perfect-fit line |
| `plot_feature_importance(importances, names)` | Horizontal bar chart |

---

## 📄 Dataset Format

Your CSV must contain at least:

| Column | Description |
|---|---|
| `generated_power_kw` | **Target** — solar power output in kilowatts |
| `temperature_2_m_above_gnd` | Air temperature at 2 m (°C) |
| `shortwave_radiation_backwards_sfc` | Solar irradiance (W/m²) |
| `zenith` | Solar zenith angle (degrees) |
| `azimuth` | Solar azimuth angle (degrees) |
| *(+ other weather columns listed above)* | |

The bundled `data/solar_data.csv` is a ready-to-use example.

---

## 🔬 Notebooks

`notebooks/eda.ipynb` — Exploratory Data Analysis notebook for initial data investigation.

---

## 📚 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, scaling, metrics |
| `plotly` | Interactive charts |
| `matplotlib` | Supporting visualisations |

---

## 📝 License

This project is for academic / research purposes. See repository for license details.