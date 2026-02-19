import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Feature columns
TARGET = "generated_power_kw"

# Weather / sensor features 
RAW_FEATURES = [
    "temperature_2_m_above_gnd",
    "relative_humidity_2_m_above_gnd",
    "mean_sea_level_pressure_MSL",
    "total_precipitation_sfc",
    "snowfall_amount_sfc",
    "total_cloud_cover_sfc",
    "high_cloud_cover_high_cld_lay",
    "medium_cloud_cover_mid_cld_lay",
    "low_cloud_cover_low_cld_lay",
    "shortwave_radiation_backwards_sfc",
    "wind_speed_10_m_above_gnd",
    "wind_direction_10_m_above_gnd",
    "wind_speed_80_m_above_gnd",
    "wind_direction_80_m_above_gnd",
    "wind_speed_900_mb",
    "wind_direction_900_mb",
    "wind_gust_10_m_above_gnd",
    "angle_of_incidence",
    "zenith",
    "azimuth",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Basic cleaning:
    
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


# Time feature
def add_time_features(df: pd.DataFrame, readings_per_day: int = 10) -> pd.DataFrame:
    
    df = df.copy()
    n = len(df)
    df["hour_of_day"] = np.arange(n) % readings_per_day
    df["day_index"] = np.arange(n) // readings_per_day

    # Map day_index → season (approximate, 365-day cycle)
    day_of_year = df["day_index"] % 365
    conditions = [
        (day_of_year < 91),                          # Winter (Jan-Mar)
        (day_of_year >= 91) & (day_of_year < 182),   # Spring (Apr-Jun)
        (day_of_year >= 182) & (day_of_year < 273),  # Summer (Jul-Sep)
        (day_of_year >= 273),                         # Autumn (Oct-Dec)
    ]
    season_labels = [0, 1, 2, 3]
    df["season"] = np.select(conditions, season_labels, default=0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    # List of feature columns available
    engineered = ["hour_of_day", "day_index", "season"]
    features = [c for c in RAW_FEATURES if c in df.columns]
    features += [c for c in engineered if c in df.columns]
    return features