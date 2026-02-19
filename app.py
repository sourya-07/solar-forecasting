import os
import sys
import numpy as np
import pandas as pd
import streamlit as st


sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_csv, load_uploaded_file
from src.preprocessing import clean_data, add_time_features, prepare_data, TARGET
from src.model import (
    train_linear_regression,
    train_random_forest,
    predict,
    evaluate_model,
)
from src.visualization import (
    plot_power_trend,
    plot_hourly_pattern,
    plot_seasonal_pattern,
    plot_actual_vs_predicted,
    plot_scatter_actual_vs_predicted,
    plot_feature_importance,
    plot_correlation_heatmap,
)

# Page config
st.set_page_config(
    page_title="Solar Energy Forecasting",
    page_icon="☀️",
    layout="wide",
)

# Sidebar data source
st.sidebar.title("☀️ Solar Forecasting")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data source",
    ["Default dataset", "Upload CSV"],
)

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "solar_data.csv")

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        raw_df = load_uploaded_file(uploaded)
    else:
        st.info("⬆️ Please upload a CSV file to get started.")
        st.stop()
else:
    raw_df = load_csv(DEFAULT_CSV)

# Preprocess once and cache
df_clean = clean_data(raw_df)
df_featured = add_time_features(df_clean)

# Tabs section
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Trends & Seasonality", "Forecasting"])