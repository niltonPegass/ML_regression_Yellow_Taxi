import os
import pandas as pd
from src import config


def resolve_data_path(file_path: str = None) -> str:
    """Resolve the CSV path, preferring an explicit path, then local data, then Kaggle."""
    if file_path:
        return file_path
    if os.path.exists(config.DATA_PATH):
        return config.DATA_PATH
    return config.DEFAULT_KAGGLE_DATA_PATH


def load_data(file_path: str = None, nrows: int = None) -> pd.DataFrame:
    """Loads the yellow taxi dataset from the configured CSV path."""
    file_path = resolve_data_path(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            f"Place it at {config.DATA_PATH} or pass a valid file_path."
        )

    sample_head = pd.read_csv(file_path, nrows=5)
    print("Colunas detectadas:\n", sample_head.columns.tolist())
    print(f"Loading dataset from: {file_path}")
    return pd.read_csv(file_path, nrows=nrows)


def display_dataset_overview(df: pd.DataFrame) -> None:
    """Display dataset shape, a sample, info, and descriptive statistics."""
    print(
        f">> Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns "
        f"({df.shape[0] * df.shape[1]} data points)\n"
    )

    print(">> First 5 rows:")
    print(df.head(5).to_string())

    print("\n>> General dataset information:")
    df.info()

    overview_columns = ['trip_distance', 'fare_amount', 'extra', 'tip_amount']
    existing_columns = [col for col in overview_columns if col in df.columns]
    print("\n>> Descriptive statistics:")
    print(df[existing_columns].describe().to_string())
