#!/usr/bin/env python3
"""
baseline_pipelines.py

Entrena dos pipelines baseline para regresión salarial usando únicamente
las columnas numéricas de tu dataset (skill_*, other_*...).
Convierte rangos de salario a float antes de entrenar.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("baseline")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()


def parse_salary(s: str) -> float:
    """
    Convierte un string de rango salarial tipo '$90,000 - $110,000'
    a la media numérica.
    """
    if pd.isna(s):
        return 0.0
    # quitar símbolo y comas
    txt = s.replace('$', '').replace(',', '')
    parts = txt.split('-')
    try:
        nums = [float(p.strip()) for p in parts]
        return sum(nums) / len(nums)
    except ValueError:
        # si no es un rango, intenta float directo
        return float(txt.strip())


def build_column_transformer(numeric_cols) -> ColumnTransformer:
    """
    Construye un ColumnTransformer que:
      - Imputa (mediana) columnas numéricas.
      - Escala (StandardScaler) columnas numéricas.
    """
    return ColumnTransformer([
        ('imputer', SimpleImputer(strategy='median'), numeric_cols),
        ('scaler', StandardScaler(), numeric_cols),
    ], remainder='drop')


def build_pipelines(transformer: ColumnTransformer):
    """
    Retorna pipelines A y B:
      A: imputación+escalado + regresión lineal
      B: imputación+escalado + polinomio(deg=2) + regresión lineal
    """
    pipe_a = Pipeline([
        ('preproc', transformer),
        ('model', LinearRegression())
    ])
    pipe_b = Pipeline([
        ('preproc', transformer),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', LinearRegression())
    ])
    return {'A': pipe_a, 'B': pipe_b}


def evaluate_model(pipe, X, y) -> dict:
    preds = pipe.predict(X)
    return {
        'mae': mean_absolute_error(y, preds),
        'mse': mean_squared_error(y, preds),
        'r2': r2_score(y, preds),
        'mape': (abs((y - preds) / y).mean()) * 100
    }


def train_and_serialize(config: dict) -> None:
    # 1) Carga de datos
    df_train = pd.read_csv(config['data']['train_path'])
    df_hold = pd.read_csv(config['data']['holdout_path'])
    target = config['target']

    # 2) Convertir target a float
    df_train[target] = df_train[target].apply(parse_salary)
    df_hold[target] = df_hold[target].apply(parse_salary)

    # 3) Separar X/y
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_hold = df_hold.drop(columns=[target])
    y_hold = df_hold[target]

    # 4) Inferencia de numéricas si lista vacía
    if not config['transformer']['numeric_cols']:
        numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
        logger.info("Columnas numéricas detectadas: %s", numeric_cols)
    else:
        numeric_cols = config['transformer']['numeric_cols']

    # 5) Construir transformador y pipelines
    transformer = build_column_transformer(numeric_cols)
    pipelines = build_pipelines(transformer)

    # 6) Crear directorio de salida
    outdir = Path(config['baseline']['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    # 7) Entrenar, serializar y evaluar
    metrics_summary = {}
    for name, pipe in pipelines.items():
        logger.info(f"Entrenando pipeline {name}")
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, outdir / f"pipeline_{name}.pkl")
        logger.info(f"Pipeline {name} guardado.")

        logger.info(f"Evaluando pipeline {name}")
        metrics = evaluate_model(pipe, X_hold, y_hold)
        metrics_summary[name] = metrics
        (outdir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"Métricas {name}: {metrics}")

    # 8) Guardar resumen
    (outdir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))
    logger.info("Resumen de métricas guardado.")


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline pipelines regresión salarial")
    parser.add_argument('--config','-c', required=True, help='Ruta a YAML/JSON config')
    return parser.parse_args()


def load_config(path: str) -> dict:
    text = Path(path).read_text(encoding='utf-8')
    if path.lower().endswith(('.yaml','.yml')):
        return yaml.safe_load(text)
    return json.loads(text)


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args.config)
    train_and_serialize(cfg)
