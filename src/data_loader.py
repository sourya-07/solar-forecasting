import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return df