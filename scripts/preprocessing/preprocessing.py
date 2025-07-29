#!/usr/bin/env python3
"""
baseline_pipelines.py

Fase 1: Baseline extendido que incluye texto (TF-IDF) y variables numéricas.
Entrena dos pipelines (A: LinearRegression, B: PolynomialFeatures + LinearRegression),
serializa modelos y guarda métricas.
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
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
    Convierte un string de rango salarial tipo '$90,000 - $110,000' a la media numérica.
    """
    if pd.isna(s):
        return 0.0
    txt = s.replace('$','').replace(',','')
    parts = txt.split('-')
    try:
        nums = [float(p.strip()) for p in parts]
        return sum(nums)/len(nums)
    except:
        try:
            return float(txt.strip())
        except:
            return 0.0


def build_column_transformer(numeric_cols, cat_cols, text_cols, tfidf_max_features=1000) -> ColumnTransformer:
    """
    Construye un ColumnTransformer que:
      - Imputa y escala numéricas.
      - Aplica OneHot a categóricas.
      - Aplica TF-IDF a texto.
    """
    transformers = []
    if numeric_cols:
        transformers.append(('impute_num', SimpleImputer(strategy='median'), numeric_cols))
        transformers.append(('scale_num', StandardScaler(), numeric_cols))
    if cat_cols:
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    if text_cols:
        # aplanar columna de texto y vectorizar
        transformers.append(
            ('tfidf', Pipeline([
                ('flatten', FunctionTransformer(lambda X: X.ravel(), validate=False)),
                ('vect', TfidfVectorizer(max_features=tfidf_max_features))
            ]), text_cols)
        )
    return ColumnTransformer(transformers, remainder='drop')


def build_pipelines(transformer: ColumnTransformer):
    pipe_a = Pipeline([('preproc', transformer), ('model', LinearRegression())])
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
        'mape': (abs((y-preds)/y).mean())*100
    }


def train_and_serialize(config: dict) -> None:
    # carga CSVs
    df_train = pd.read_csv(config['data']['train_path'])
    df_hold = pd.read_csv(config['data']['holdout_path'])
    target = config['target']

    # parse salary\ n    df_train[target] = df_train[target].apply(parse_salary)
    df_hold[target] = df_hold[target].apply(parse_salary)

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    X_hold = df_hold.drop(columns=[target])
    y_hold = df_hold[target]

    # columnas\ n    numeric_cols = config['transformer']['numeric_cols'] or X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = config['transformer']['categorical_cols']
    text_cols = config['transformer']['text_cols']

    logger.info(f"Num cols: {numeric_cols}, Cat cols: {cat_cols}, Text cols: {text_cols}")

    transformer = build_column_transformer(numeric_cols, cat_cols, text_cols)
    pipelines = build_pipelines(transformer)

    outdir = Path(config['baseline']['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_summary = {}
    for name, pipe in pipelines.items():
        logger.info(f"Training {name}")
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, outdir / f"pipeline_{name}.pkl")
        logger.info(f"Saved pipeline {name}")
        metrics = evaluate_model(pipe, X_hold, y_hold)
        metrics_summary[name] = metrics
        (outdir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"Metrics {name}: {metrics}")

    (outdir / 'metrics_summary.json').write_text(json.dumps(metrics_summary, indent=2))
    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline with text TF-IDF")
    parser.add_argument('--config','-c', required=True, help='YAML/JSON config')
    return parser.parse_args()


def load_config(path: str) -> dict:
    txt = Path(path).read_text(encoding='utf-8')
    return yaml.safe_load(txt) if path.endswith(('.yaml','.yml')) else json.loads(txt)

if __name__=='__main__':
    args = parse_args()
    cfg = load_config(args.config)
    train_and_serialize(cfg)
