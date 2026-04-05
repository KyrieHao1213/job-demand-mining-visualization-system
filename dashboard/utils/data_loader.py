from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from config import DATASET_REGISTRY


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all_datasets() -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for name, path in DATASET_REGISTRY.items():
        datasets[name] = load_csv(path)
    return datasets


def get_dataset(datasets: Dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    return datasets.get(key, pd.DataFrame()).copy()


def require_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f'{dataset_name} 缺少字段: {missing}')
