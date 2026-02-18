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
