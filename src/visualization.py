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