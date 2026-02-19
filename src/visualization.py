import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Trend
def plot_power_trend(df: pd.DataFrame):
    # Line plot of solar power generation over time
    fig = px.line(
        df.reset_index(),
        x="index",
        y="generated_power_kw",
        title="Solar Power Generation Trend",
        labels={"index": "Time Step", "generated_power_kw": "Power (kW)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


# Seasonality

def plot_hourly_pattern(df: pd.DataFrame):
    """Box plot of power generation grouped by hour_of_day."""
    if "hour_of_day" not in df.columns:
        return _empty_figure("hour_of_day column missing")

    fig = px.box(
        df,
        x="hour_of_day",
        y="generated_power_kw",
        title="Power Generation by Hour of Day",
        labels={"hour_of_day": "Hour of Day", "generated_power_kw": "Power (kW)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_seasonal_pattern(df: pd.DataFrame):
    """Box plot of power generation grouped by season."""
    if "season" not in df.columns:
        return _empty_figure("season column missing")

    season_map = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
    temp = df.copy()
    temp["season_name"] = temp["season"].map(season_map)

    fig = px.box(
        temp,
        x="season_name",
        y="generated_power_kw",
        title="Power Generation by Season",
        labels={"season_name": "Season", "generated_power_kw": "Power (kW)"},
        category_orders={"season_name": ["Winter", "Spring", "Summer", "Autumn"]},
    )
    fig.update_layout(template="plotly_white")
    return fig

