#!/usr/bin/env python3
"""
split_data.py

Divide un dataset en train y hold-out, y guarda los CSVs correspondientes.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuración de logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    ))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error("No existe el dataset en %s", path)
        raise FileNotFoundError(f"{path} no encontrado")
    df = pd.read_csv(path)
    logger.info("Dataset cargado: %d filas, %d columnas", *df.shape)
    return df

def split_and_save(
    df: pd.DataFrame,
    target_col: str,
    train_path: Path,
    holdout_path: Path,
    test_size: float = 0.2,
    random_state: int = 42
):
    if target_col not in df.columns:
        logger.error("Columna objetivo '%s' no está en el dataset", target_col)
        raise KeyError(f"{target_col} missing")
    train_df, holdout_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    # Guardar
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    logger.info("Guardado train (%d filas) en %s", len(train_df), train_path)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(holdout_path, index=False)
    logger.info("Guardado hold-out (%d filas) en %s", len(holdout_df), holdout_path)

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[1]
    raw_path = BASE / "data" / "ofertas_variables_semanticas.csv"  # ajusta si tu nombre es otro
    train_path = BASE / "data" / "splits" / "train.csv"
    holdout_path = BASE / "data" / "splits" / "holdout.csv"

    df = load_dataset(raw_path)
    split_and_save(
    df,
    target_col="salary",
    train_path=train_path,
    holdout_path=holdout_path,
    test_size=0.2,
    random_state=42
)
