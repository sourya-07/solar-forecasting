"""Data loading utilities for the Solar Forecasting app."""

import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file from disk and return it as a DataFrame.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        pandas DataFrame with the raw data.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    except Exception as e:
        raise ValueError(f"Could not parse CSV file '{filepath}': {e}") from e


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Parse a Streamlit UploadedFile object and return it as a DataFrame.

    Args:
        uploaded_file: A file-like object returned by st.file_uploader.

    Returns:
        pandas DataFrame with the uploaded data.

    Raises:
        ValueError: If the uploaded content cannot be parsed as CSV.
    """
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not parse the uploaded file: {e}") from e