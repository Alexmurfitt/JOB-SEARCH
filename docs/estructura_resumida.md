# Informe de la estructura del proyecto

# .

- **05_modeling.py** (1.5KB)
  ```
# scripts/05_modeling.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Rutas
INPUT_PATH = "data/ofertas_variables_semanticas.csv"
OUTPUT_PATH = "models/salary_predictor.pkl"
os.makedirs("models", exist_ok=True)

# 1. Leer archivo
df = pd.read_csv(INPUT_PATH)

# 2. Preprocesar salario → media del rango
def parse_salary(s):
    try:
        parts = s.replace("$", "").replace(",", "").split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        elif len(parts) == 1:
            return float(parts[0])
    except:
        return np.nan

df["salary_num"] = df["salary"].apply(parse_salary)
df = df.dropna(subset=["salary_num"])

# 3. Seleccionar características y target
X = df.drop(columns=["salary", "title", "description", "salary_num"])
y = df["salary_num"]

# 4. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluar
y_pred = model.predict(X_test)
print(f"📊 MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"📈 R2: {r2_score(y_test, y_pred):.4f}")

# 6. Guardar modelo
joblib.dump(model, OUTPUT_PATH)
print(f"✅ Modelo guardado en: {OUTPUT_PATH}")
  ```
- **06_modeling_optimizado.py** (2.3KB)
  ```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 📥 Cargar el dataset
print("📥 Cargando datos...")
df = pd.read_csv("data/ofertas_variables_semanticas_numerico.csv")

# 🧹 Preprocesamiento
df = df.dropna(subset=["salary_numeric"])
X = df.drop(columns=["salary", "salary_numeric", "title", "description"])
y = df["salary_numeric"]

# 🔀 División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Entrenar modelo con XGBoost
print("⚙️ Entrenando modelo XGBoost optimizado...")
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=4,
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 📈 Evaluación
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"📊 MAE: {mae:,.2f}")
print(f"📈 R²: {r2:.4f}")
print(f"📉 RMSE: {rmse:,.2f}")

# 🧪 Importancia de características
print("📊 Generando gráfico de importancia de variables...")
importancias = model.feature_importances_
features = X.columns
df_importancia = pd.DataFrame({
    "Característica": features,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

# 🎨 Gráfico
plt.figure(figsize=(10, 6))
plt.barh(df_importancia["Característica"], df_importancia["Importancia"], color="skyblue")
plt.xlabel("Importancia")
plt.ylabel("Característica")
plt.title("Importancia de las características para predecir el salario")
plt.gca().invert_yaxis()
plt.tight_layout()

# 💾 Guardar gráfico
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/importancia_variables_modelo_final.png")
print("✅ Gráfico guardado en: reports/importancia_variables_modelo_final.png")

# 💾 Guardar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_predictor.pkl")
print("✅ Modelo guardado en: models/salary_predictor.pkl")
  ```
- **advanced_pipelines.py** (2.6KB)
  ```
#!/usr/bin/env python3
"""Pipeline de predicción de salario."""
import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
# Aquí otros imports necesarios: sklearn.compose, vectorizers, modelos...

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def detect_input_path(data_cfg):
    splits_dir = Path(data_cfg["splits_dir"])
    train_path = splits_dir / "train.csv"
    holdout_path = splits_dir / data_cfg["holdout"]
    return train_path, holdout_path

def parse_salary(salary_str):
    parts = salary_str.replace("$","").split("-")
    nums = [int(p) for p in parts if p.isdigit() or p.isdecimal()]
    return sum(nums)/len(nums)

def load_data(train_path, holdout_path, target):
    df_train = pd.read_csv(train_path)
    df_hold = pd.read_csv(holdout_path)
    y_train = df_train[target].apply(parse_salary)
    y_hold = df_hold[target].apply(parse_salary)
    X_train = df_train.drop(columns=[target])
    X_hold = df_hold.drop(columns=[target])
    return X_train, X_hold, y_train, y_hold

def build_pipelines(cfg):
    pipelines = {}
    # TODO: definir ColumnTransformer para columnas numéricas y textuales
    #       y crear pipelines para cada modelo configurado en cfg["model"]
    return pipelines

def evaluate_and_save(pipelines, X_train, X_hold, y_train, y_hold, artifacts_cfg):
    results_train = {}
    results_hold = {}
    for name, pipe in pipelines.items():
        logging.info(f"Entrenando {name}...")
        pipe.fit(X_train, y_train)
        # TODO: calcular métricas mae, mse, r2 y rellenar results_train[name], results_hold[name]
    metrics_dir = Path(artifacts_cfg["metrics_path"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics_train.json").write_text(yaml.dump(results_train))
    (metrics_dir / "metrics_holdout.json").write_text(yaml.dump(results_hold))
    logging.info("Métricas guardadas en artifacts/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train_path, holdout_path = detect_input_path(cfg["data"])
    X_train, X_hold, y_train, y_hold = load_data(train_path, holdout_path, args.target)
    pipelines = build_pipelines(cfg)
    evaluate_and_save(pipelines, X_train, X_hold, y_train, y_hold, cfg["artifacts"])

if __name__ == "__main__":
    main()
  ```
- **analysis_and_modeling_advanced.py** (5.4KB)
  ```
import pandas as pd
import numpy as np
import shap
import joblib
import optuna
import argparse
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from fpdf import FPDF

warnings.filterwarnings("ignore")

# ==============================
# Argumentos CLI
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--salary-column", required=True)
parser.add_argument("--output-dir", default="models")
parser.add_argument("--report", default="reports/informe_modelo_salarial.pdf")
parser.add_argument("--force-simple-model", action="store_true")
args = parser.parse_args()

# ==============================
# Cargar datos
# ==============================
print("\U0001F4E5 Cargando dataset de features...")
df = pd.read_csv(args.input)
if len(df) < 10:
    print("⚠️ Dataset demasiado pequeño para modelos robustos (solo {} filas).".format(len(df)))

# ==============================
# Features y target
# ==============================
# Eliminar columnas no numéricas y problemáticas explícitas
drop_cols = [args.salary_column, "salary", "title", "description"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[args.salary_column]

# ==============================
# Reducir dimensionalidad
# ==============================
k = min(5, X.shape[1])
# X ya está listo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==============================
# Optuna para XGBoost
# ==============================
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.2),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

print("\U0001F50D Optimizando XGBoost con Optuna...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, timeout=600)
best_params = study.best_params
print(f"✅ Mejores parámetros XGB: {best_params}")

xgb_model = XGBRegressor(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)

# ==============================
# Modelos adicionales y ensemble
# ==============================
lgb_model = LGBMRegressor(random_state=42)
cat_model = CatBoostRegressor(silent=True, random_state=42)
stacking = StackingRegressor(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model), ("cat", cat_model)],
    final_estimator=LinearRegression(),
)
stacking.fit(X_train, y_train)
ensemble_preds = stacking.predict(X_test)
mae_ensemble = mean_absolute_error(y_test, ensemble_preds)
r2_ensemble = r2_score(y_test, ensemble_preds)
print(f"✅ Ensemble — MAE: {mae_ensemble:.2f}, R²: {r2_ensemble:.2f}")

# ==============================
# Modelo simple: Linear Regression
# ==============================
simple_model = LinearRegression()
simple_model.fit(X_train, y_train)
simple_preds = simple_model.predict(X_test)
simple_mae = mean_absolute_error(y_test, simple_preds)
simple_r2 = r2_score(y_test, simple_preds)

# ==============================
# Comparación gráfica
# ==============================
fig, ax = plt.subplots()
models = ["Linear", "Ensemble"]
maes = [simple_mae, mae_ensemble]
ax.bar(models, maes)
ax.set_title("Comparación de MAE")
ax.set_ylabel("MAE")
plt.tight_layout()
plt.savefig("mae_comparison.png")

# ==============================
# SHAP explicabilidad
# ==============================
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")

# ==============================
# Generar informe PDF
# ==============================
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Informe de Modelo Salarial", ln=1, align="C")
pdf.ln(10)

pdf.multi_cell(0, 10, f"Tamaño del dataset: {len(df)} registros")
if len(df) < 10:
    pdf.set_text_color(255, 0, 0)
    pdf.multi_cell(0, 10, "⚠️ Advertencia: dataset demasiado pequeño para entrenamiento robusto")
    pdf.set_text_color(0, 0, 0)

pdf.multi_cell(0, 10, f"MAE Ensemble: {mae_ensemble:.2f} | R²: {r2_ensemble:.2f}")
pdf.multi_cell(0, 10, f"MAE Linear: {simple_mae:.2f} | R²: {simple_r2:.2f}")
pdf.image("mae_comparison.png", w=160)
pdf.image("shap_summary.png", w=160)

pdf.output(args.report)
print(f"📄 Informe generado en: {args.report}")

# ==============================
# Guardar modelo
# ==============================
joblib.dump(stacking, f"{args.output_dir}/salary_predictor_ensemble.pkl")
joblib.dump(simple_model, f"{args.output_dir}/salary_predictor_linear.pkl")
print("✅ Modelos guardados correctamente.")
  ```
- **analyze_and_filter_correlations.py** (3.5KB)
  ```

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main(input_csv, output_csv, report_path, heatmap_path, salary_column='salary_numeric', low_threshold=0.1, high_threshold=0.9):
    print("📥 Cargando archivo...")
    df = pd.read_csv(input_csv)

    df_num = df.select_dtypes(include=np.number)
    if salary_column not in df_num.columns:
        raise ValueError(f"❌ La columna '{salary_column}' no está en el CSV.")

    print("📊 Calculando matriz de correlaciones...")
    corr = df_num.corr()

    # Variables poco correlacionadas con salario
    low_corr_features = corr[salary_column][abs(corr[salary_column]) < low_threshold].index.tolist()
    if salary_column in low_corr_features:
        low_corr_features.remove(salary_column)

    # Variables redundantes entre sí
    redundant_pairs = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > high_threshold:
                col1 = corr.columns[i]
                col2 = corr.columns[j]
                redundant_pairs.add((col1, col2))

    redundant_to_remove = set([b for a, b in redundant_pairs if a != salary_column])
    to_drop = sorted(set(low_corr_features).union(redundant_to_remove))
    df_filtered = df.drop(columns=to_drop)

    # Guardar resultados
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🧠 Análisis de correlaciones automáticas\n")
        f.write(f"Archivo de entrada: {input_csv}\n\n")
        f.write("Variables poco correlacionadas con salario (|corr| < 0.1):\n")
        for var in low_corr_features:
            f.write(f"  - {var}: {corr[salary_column][var]:.3f}\n")

        f.write("\nVariables redundantes (|corr| > 0.9 entre sí):\n")
        for a, b in sorted(redundant_pairs):
            f.write(f"  - {a} ~ {b}: {corr.loc[a, b]:.3f}\n")

        f.write(f"\nTotal de variables eliminadas: {len(to_drop)}\n")
        f.write("Columnas eliminadas: {}\n".format(', '.join(to_drop)))
        f.write(f"\n✅ Archivo final sin columnas irrelevantes: {output_csv}")

    print(f"✅ Archivo limpio guardado: {output_csv}")
    print(f"📝 Informe generado: {report_path}")

    # Generar heatmap
    print("🖼️ Generando heatmap de correlaciones...")
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Heatmap de Correlaciones", fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    plt.savefig(heatmap_path)
    print(f"✅ Heatmap guardado como: {heatmap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", default="data/ofertas_filtradas.csv", help="CSV limpio de salida")
    parser.add_argument("--report", default="reports/informe_correlaciones.txt", help="Informe de correlaciones")
    parser.add_argument("--heatmap", default="reports/heatmap_correlaciones.png", help="Imagen de salida del heatmap")
    parser.add_argument("--salary-column", default="salary_numeric", help="Nombre de la columna objetivo")
    args = parser.parse_args()

    main(args.input, args.output, args.report, args.heatmap, args.salary_column)
  ```
- **analyze_semantic_entities.py** (2.0KB)
  ```
# scripts/analyze_semantic_entities.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
from ast import literal_eval

INPUT_FILE = "data/entities_with_semantic_classification.csv"
OUTPUT_DIR = "reports/semantic_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar datos
df = pd.read_csv(INPUT_FILE)

# Asegurar que la columna de clasificación está como lista de tuplas (si se guardó como string)
if isinstance(df.loc[0, 'semantic_classification'], str):
    df['semantic_classification'] = df['semantic_classification'].apply(literal_eval)

# Aplanar todas las tuplas [(entidad, clase), ...]
all_pairs = [pair for sublist in df['semantic_classification'] for pair in sublist]
entities, classes = zip(*all_pairs)

# Frecuencia total por clase
df_stats = pd.DataFrame({'entity': entities, 'class': classes})
class_counts = df_stats['class'].value_counts()
print("\n📊 Frecuencia por clase semántica:\n", class_counts)

# Guardar como CSV
class_counts.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))

# Top 10 entidades por clase
top_entities_by_class = {}
for cls in df_stats['class'].unique():
    top = df_stats[df_stats['class'] == cls]['entity'].value_counts().head(10)
    top.to_csv(os.path.join(OUTPUT_DIR, f"top10_{cls}.csv"))
    top_entities_by_class[cls] = top

# Gráfico general por clase
plt.figure(figsize=(10, 6))
class_counts.plot(kind='bar', title='Frecuencia por tipo de entidad')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "frecuencia_por_clase.png"))
plt.show()

# Gráficos individuales por clase (top 10 entidades)
for cls, series in top_entities_by_class.items():
    plt.figure(figsize=(10, 6))
    series.plot(kind='bar', title=f'Top 10 {cls}')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"top10_{cls}.png"))
    plt.close()

print(f"\n✅ Análisis completo guardado en: {OUTPUT_DIR}")
  ```
- **augment_and_analyze.py** (1.7KB)
  ```
import os
import sys
import subprocess
import argparse


def run_augmentation(input_path, output_path, min_rows=100):
    print("🧪 Ejecutando aumento de datos sintéticos...")
    cmd = [
        sys.executable, "scripts/augment_data_with_synthetic.py",
        "--input", input_path,
        "--output", output_path,
        "--min-rows", str(min_rows)
    ]
    subprocess.run(cmd, check=True)


def run_analysis(input_path, salary_column, output_dir, report_path):
    print("📊 Ejecutando análisis y modelado...")
    cmd = [
        sys.executable, "scripts/analysis_and_modeling_advanced.py",
        "--input", input_path,
        "--salary-column", salary_column,
        "--output-dir", output_dir,
        "--report", report_path
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Archivo CSV original con datos numerizados")
    parser.add_argument("--output", default="data/ofertas_variables_semanticas_numerico_augmented.csv", help="Ruta donde guardar el CSV aumentado")
    parser.add_argument("--salary-column", default="salary_numeric", help="Nombre de la columna de salario")
    parser.add_argument("--output-dir", default="models", help="Directorio para guardar el modelo")
    parser.add_argument("--report", default="reports/informe_modelo_augmented.pdf", help="Ruta del informe PDF")
    parser.add_argument("--min-rows", type=int, default=100, help="Mínimo de filas deseadas")
    args = parser.parse_args()

    run_augmentation(args.input, args.output, args.min_rows)
    run_analysis(args.output, args.salary_column, args.output_dir, args.report)


if __name__ == "__main__":
    main()
  ```
- **augment_data_with_synthetic.py** (1.9KB)
  ```
import pandas as pd
import numpy as np
import argparse
import os

def parse_salary(salary_str):
    """Convierte una cadena como '$90,000 - $110,000' a la media numérica."""
    if pd.isna(salary_str):
        return np.nan
    try:
        salary_str = salary_str.replace("$", "").replace(",", "")
        parts = salary_str.split("-")
        numbers = [float(p.strip()) for p in parts]
        return sum(numbers) / len(numbers) if numbers else np.nan
    except Exception:
        return np.nan

def generate_synthetic_row(reference_row):
    """Genera una nueva fila basada en una de ejemplo (con pequeñas variaciones)."""
    new_row = reference_row.copy()
    for col in reference_row.index:
        if isinstance(reference_row[col], (int, float)):
            noise = np.random.normal(0, 1)
            new_row[col] = reference_row[col] + noise
    return new_row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-rows", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"📊 Datos reales: {len(df)} filas")

    # Aseguramos que salary_numeric exista
    if "salary_numeric" not in df.columns:
        df["salary_numeric"] = df["salary"].apply(parse_salary)

    rows_needed = args.min_rows - len(df)
    if rows_needed > 0:
        print(f"⚠️ Dataset incompleto. Generando {rows_needed} datos sintéticos...")
        synthetic_rows = [generate_synthetic_row(df.sample(1).iloc[0]) for _ in range(rows_needed)]
        df_synthetic = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, df_synthetic], ignore_index=True)

    df.to_csv(args.output, index=False)
    print(f"✅ Dataset aumentado guardado en: {args.output} (total: {len(df)} filas)")

if __name__ == "__main__":
    main()
  ```
- **baseline_pipelines.py** (5.2KB)
  ```
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
  ```
- **clean_semantic_labels.py** (2.5KB)
  ```
import pandas as pd
import os

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
INPUT_FILE = "data/final_entities_semantic.csv"
OUTPUT_FILE = "data/final_entities_semantic_cleaned.csv"

# Diccionario de equivalencias semánticas
SEMANTIC_EQUIVALENCES = {
    "powerbi": "power bi",
    "power bi": "power bi",
    "python3": "python",
    "python 3": "python",
    "py": "python",
    "excel avanzado": "excel",
    "microsoft excel": "excel",
    "sql server": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "tableau software": "tableau",
    "data analyst": "data analysis",
    "data analytics": "data analysis",
    "scikit-learn": "sklearn",
    "sklearn": "sklearn",
    "tensorflow 2": "tensorflow",
    "keras": "tensorflow",
    "licenciatura": "bachelor's degree",
    "grado": "bachelor's degree",
    "máster": "master's degree",
    "maestría": "master's degree",
    "doctorado": "phd",
    "ph.d.": "phd",
    "english": "english",
    "ingles": "english",
    "b2": "intermediate",
    "c1": "advanced",
    "senior": "seniority_senior",
    "jr": "seniority_junior",
    "junior": "seniority_junior",
    "mid": "seniority_mid",
    "medio": "seniority_mid",
    "machine learning": "ml",
    "ml": "ml"
}

def normalize(text):
    return str(text).strip().lower()

def unify_entity(entity):
    norm_entity = normalize(entity)
    return SEMANTIC_EQUIVALENCES.get(norm_entity, norm_entity)

# ===============================
# EJECUCIÓN
# ===============================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Archivo no encontrado: {INPUT_FILE}")
        return

    print(f"📥 Cargando datos desde {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    if not {"description_id", "entity", "semantic_class"}.issubset(df.columns):
        print("❌ Columnas requeridas no encontradas en el archivo CSV.")
        print(f"🔎 Columnas detectadas: {list(df.columns)}")
        return

    print("🧹 Aplicando limpieza semántica avanzada...")
    df["entity_cleaned"] = df["entity"].apply(unify_entity)

    columnas_finales = ["description_id", "entity", "entity_cleaned", "semantic_class"]
    df = df[columnas_finales]

    print(f"💾 Guardando resultado limpio en {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Limpieza semántica completada.")

if __name__ == "__main__":
    main()
  ```
- **convertir_salario_numerico.py** (1.2KB)
  ```
import pandas as pd
import os
import re

# Ruta de entrada y salida
INPUT_PATH = "data/ofertas_variables_semanticas.csv"
OUTPUT_PATH = "data/ofertas_variables_semanticas_numerico.csv"

def convertir_rango_salarial(s):
    if pd.isna(s):
        return None
    match = re.findall(r'\$?([\d,]+)', s)
    if match:
        nums = [int(x.replace(',', '')) for x in match]
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return sum(nums) / 2
    return None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Archivo no encontrado: {INPUT_PATH}")
        return

    print(f"📥 Leyendo archivo: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    if 'salary' not in df.columns:
        print("❌ Columna 'salary' no encontrada.")
        return

    print("🔄 Convirtiendo rangos salariales a valores numéricos...")
    df['salary_numeric'] = df['salary'].apply(convertir_rango_salarial)

    print(f"💾 Guardando archivo con nueva columna en: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Proceso completado correctamente.")

if __name__ == "__main__":
    main()
  ```
- **dashboard.py** (3.2KB)
  ```
import streamlit as st

# ⚠️ Esta línea debe ser la primera de Streamlit
st.set_page_config(page_title="Análisis de Salario en Ciencia de Datos", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# CONFIGURACIÓN GENERAL
# ============================
STRUCTURED_FILE = "data/ofertas_variables_semanticas.csv"
RAW_FILE = "data/raw/sample_jobs.csv"

# ============================
# CARGA DE DATOS
# ============================
@st.cache_data
def load_data():
    df_structured = pd.read_csv(STRUCTURED_FILE)
    df_raw = pd.read_csv(RAW_FILE)
    df_structured["salary"] = df_raw["salary"]  # Asignación por índice

    def parse_salary(s):
        if pd.isna(s): return None
        s = str(s).replace("$", "").replace(",", "").strip()
        if "-" in s:
            try:
                low, high = s.split("-")
                return (float(low) + float(high)) / 2
            except:
                return None
        try:
            return float(s)
        except:
            return None

    df_structured["salary_numeric"] = df_structured["salary"].apply(parse_salary)
    return df_structured

df = load_data()

# ============================
# INTERFAZ STREAMLIT
# ============================
st.title("📊 Dashboard: Formación, Skills y Salario en Ciencia de Datos")
st.sidebar.header("🎯 Filtros")

skills = [col for col in df.columns if col.startswith("skill_")]
certs = [col for col in df.columns if col.startswith("cert_")]
educs = [col for col in df.columns if col.startswith("edu_")]

selected_skills = st.sidebar.multiselect("Skills", options=skills)
selected_certs = st.sidebar.multiselect("Certificaciones", options=certs)
selected_educs = st.sidebar.multiselect("Formaciones", options=educs)

filtered_df = df.copy()
for col in selected_skills + selected_certs + selected_educs:
    filtered_df = filtered_df[filtered_df[col] == 1]

st.markdown(f"🎓 Se encontraron **{len(filtered_df)}** ofertas que cumplen los filtros seleccionados.")

if len(filtered_df) > 0:
    st.subheader("💵 Distribución Salarial")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_df["salary_numeric"], kde=True, ax=ax, bins=10)
    ax.set_title("Distribución de Salarios")
    ax.set_xlabel("Salario promedio (USD)")
    st.pyplot(fig)

    def plot_top_entities(columns, title, top_n=10):
        counts = filtered_df[columns].sum().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index.str.replace("_", " "), ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Frecuencia")
        st.pyplot(fig)

    st.subheader("🏆 Skills más comunes")
    plot_top_entities(skills, "Top Skills")

    st.subheader("🎓 Certificaciones más comunes")
    plot_top_entities(certs, "Top Certificaciones")

    st.subheader("📚 Formaciones más comunes")
    plot_top_entities(educs, "Top Formaciones")

st.subheader("🧾 Datos Filtrados")
st.dataframe(filtered_df[["salary", "salary_numeric"] + selected_skills + selected_certs + selected_educs])
  ```
- **debug_check_types.py** (343.0B)
  ```
# scripts/debug_check_types.py
import pandas as pd

df = pd.read_csv("data/ofertas_variables_semanticas_numerico_augmented.csv")

print("\n📊 Columnas no numéricas (tipos de datos):")
print(df.dtypes[df.dtypes == "object"])

print("\n🔍 Valores únicos en salary_numeric (muestra):")
print(df["salary_numeric"].unique()[:10])
  ```
- **estructura_documentada_extendida.txt** (1.1MB)
  ```
# Resumen Extendido de Proyecto
**Fecha**: 2025-05-14 08:10:01
Total archivos documentados: 54

## Archivo: 05_modeling.py — 1 KB
```text
# scripts/05_modeling.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Rutas
INPUT_PATH = "data/ofertas_variables_semanticas.csv"
OUTPUT_PATH = "models/salary_predictor.pkl"
os.makedirs("models", exist_ok=True)

# 1. Leer archivo
df = pd.read_csv(INPUT_PATH)

# 2. Preprocesar salario → media del rango
def parse_salary(s):
    try:
        parts = s.replace("$", "").replace(",", "").split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        elif len(parts) == 1:
            return float(parts[0])
    except:
        return np.nan

df["salary_num"] = df["salary"].apply(parse_salary)
df = df.dropna(subset=["salary_num"])

# 3. Seleccionar características y target
X = df.drop(columns=["salary", "title", "description", "salary_num"])
y = df["salary_num"]

# 4. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluar
y_pred = model.predict(X_test)
print(f"📊 MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
... _solo primeras 50 líneas_
  ```
- **estructura_final.txt** (1.2KB)
  ```
L i s t a d o   d e   r u t a s   d e   c a r p e t a s   p a r a   e l   v o l u m e n   1 T B 
 
 E l   n  m e r o   d e   s e r i e   d e l   v o l u m e n   e s   7 C E 2 - 5 A 2 7 
 
 D : . 
 
        a n a l y s i s _ a n d _ m o d e l i n g _ a d v a n c e d . p y 
 
        a n a l y z e _ s e m a n t i c _ e n t i t i e s . p y 
 
        c l e a n _ s e m a n t i c _ l a b e l s . p y 
 
        d a s h b o a r d . p y 
 
        e s t r u c t u r a _ f i n a l . t x t 
 
        e x t r a c t _ e n t i t i e s _ t r a n s f o r m e r s . p y 
 
        g e n e r a r _ i n f o r m e . p y 
 
        g e n e r a r _ i n f o r m e _ d a t a _ a n a l y s t . p y 
 
        g e n e r a t e _ s t r u c t u r e d _ d a t a s e t . p y 
 
        l a u n c h e r . p y 
 
        m a i n _ p i p e l i n e _ j o b s . p y 
 
        n e r _ t r a n s f o r m e r s _ e x t r a c t i o n . p y 
 
        o n e n o t e _ e x t r a c t o r . p y 
 
        p r e p r o c e s s i n g _ a l l . p y 
 
        p r e p r o c e s s i n g _ n l p . p y 
 
        p r e p r o c e s s _ c s v . p y 
 
        s e m a n t i c _ c l a s s i f i e r . p y 
 
        
 
 + - - - u t i l s 
 
 
  ```
- **etl.py** (533.0B)
  ```

#!/usr/bin/env python3
# scripts/etl.py: extrae, transforma y carga los datos raw a processed

def run_etl(input_path, output_path):
    # TODO: implementar extracción, limpieza y guardado
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw', help='Carpeta raw')
    parser.add_argument('--output', default='data/processed', help='Carpeta processed')
    args = parser.parse_args()
    run_etl(args.input, args.output)
  ```
- **evaluate_model.py** (703.0B)
  ```

#!/usr/bin/env python3
# scripts/evaluate_model.py: evalúa el modelo entrenado sobre holdout

def evaluate(model_path, test_data, report_path):
    # TODO: cargar modelo, datos de test, calcular métricas y guardar reporte
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='artifacts/models/model.pkl', help='Ruta al modelo')
    parser.add_argument('--test', default='data/splits/holdout.csv', help='Datos holdout')
    parser.add_argument('--report', default='artifacts/figures', help='Dónde guardar resultados')
    args = parser.parse_args()
    evaluate(args.model, args.test, args.report)
  ```
- **extract_entities_transformers.py** (2.8KB)
  ```
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import argparse
import os

def load_model(model_name="models/ner_model"):
    """
    Carga un modelo NER fine-tuned desde un directorio local o HuggingFace.
    """
    # Si pasas un directorio local existente, lo usa; si no, intenta descargar de HF.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if (model.device.type == "cuda") else -1
    )
    return ner_pipeline

import re

def extract_entities(text, ner_pipeline):
    """
    Extrae entidades del texto, devolviendo lista de "Etiqueta:Texto" 
    y filtrando tokens vacíos o solo puntuación.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    entities = ner_pipeline(text)
    clean = []
    for ent in entities:
        word = ent["word"].strip()
        # Descartar tokens muy cortos o solo símbolos
        if len(word) <= 1 or re.fullmatch(r"[\W_]+", word):
            continue
        clean.append(f"{ent['entity_group']}:{word}")
    return clean

def process_file(input_csv, output_csv, column, model_name):
    """
    Procesa un CSV de entrada, extrae entidades usando NER y guarda un CSV enriquecido.
    """
    print(f"🔍 Cargando modelo NER desde: {model_name}")
    ner_pipeline = load_model(model_name)

    df = pd.read_csv(input_csv)
    tqdm.pandas(desc="🔎 Extrayendo entidades")
    df['extracted_entities'] = df[column].progress_apply(lambda x: extract_entities(x, ner_pipeline))

    df.to_csv(output_csv, index=False)
    print(f"💾 Guardado dataset enriquecido en: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae entidades NER de descripciones de empleo usando un modelo Transformers"
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="CSV de entrada con descripciones de empleo")
    parser.add_argument("--output", type=str, required=True,
                        help="CSV de salida con entidades extraídas")
    parser.add_argument("--column", type=str, default="description",
                        help="Nombre de la columna de texto en el CSV")
    parser.add_argument("--model", type=str, default="models/ner_model",
                        help="Ruta local (o repo HF) del modelo NER fine-tuned")
    args = parser.parse_args()

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    process_file(args.input, args.output, args.column, args.model)
  ```
- **generador_ruta_formativa.py** (3.8KB)
  ```
# generador_ruta_formativa.py

import pandas as pd
import json
import joblib
from transformers import pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from shap import TreeExplainer
import shap

# ---------------------------
# 1. Cargar modelo NER y modelos ML
# ---------------------------
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
model = joblib.load("models/salary_predictor.pkl")
shap_explainer = TreeExplainer(model)
multi_label_skills = joblib.load("models/multilabel_skills.pkl")

# ---------------------------
# 2. Definición del perfil de usuario
# ---------------------------
perfil_usuario_texto = """
Tengo 37 años, soy nativo bilingüe con doble nacionalidad (inglesa-española). He sido profesor de inglés durante 13 años.
Estudié el primer año del grado de empresariales (fundamentos de dirección, contabilidad, micro y macroeconomía, estadística, derecho, etc.).
Estudié los dos primeros años del grado de lenguas modernas (lengua, literatura, cultura inglesa, francés, narrativa audiovisual, etc.).
Tengo certificado CELTA de enseñanza de inglés a adultos.

Certificaciones:
- Oracle Autonomous Database & Machine Learning (SQL, administración, backup, recuperación, Linux, etc.).
- Curso de análisis de datos e inteligencia artificial:
  - Google Sheets, Looker Studio, MySQL
  - Python (EDA, scraping, OOP, matplotlib, etc.)
  - Machine Learning (regresión, clasificación, clustering, reducción de dimensionalidad)
  - Deep Learning (CNN, RNN, Transfer Learning)
  - Generative AI (LLMs, transformers, LangChain, RAG)
  - APIs con FastAPI, MongoDB, Docker
"""

# ---------------------------
# 3. Extraer entidades (skills) del perfil
# ---------------------------
resultados_ner = ner(perfil_usuario_texto)
skills_extraidas = list({ent['word'].lower() for ent in resultados_ner if ent['entity_group'] == 'SKILL'})
print("Skills extraídas:", skills_extraidas)

# ---------------------------
# 4. Transformación binaria
# ---------------------------
skills_binarias = multi_label_skills.transform([skills_extraidas])
skills_df = pd.DataFrame(skills_binarias, columns=multi_label_skills.classes_)

# ---------------------------
# 5. Predicción salarial actual
# ---------------------------
pred_salarial_actual = model.predict(skills_df)[0]
print(f"Salario estimado actual: {pred_salarial_actual:.2f} €")

# ---------------------------
# 6. Simulación de salario potencial añadiendo skills objetivo
# ---------------------------
skills_objetivo = [
    "python", "sql", "aws", "machine learning", "data visualization", "docker",
    "git", "cloud computing", "deep learning", "fastapi", "looker studio", "kubernetes"
]

brechas = [skill for skill in skills_objetivo if skill not in skills_extraidas]
print("Skills ausentes con alto impacto salarial:", brechas)

mejoras_df = skills_df.copy()
for skill in brechas:
    if skill in mejoras_df.columns:
        mejoras_df.at[0, skill] = 1

salario_potencial = model.predict(mejoras_df)[0]
salto_estimado = salario_potencial - pred_salarial_actual
print(f"Salario potencial si completas la formación sugerida: {salario_potencial:.2f} €")
print(f"Aumento estimado: {salto_estimado:.2f} €")

# ---------------------------
# 7. Explicabilidad con SHAP
# ---------------------------
shap_values = shap_explainer.shap_values(skills_df)
impactos = pd.DataFrame({
    'skill': skills_df.columns,
    'presente': skills_df.values[0],
    'shap_value': shap_values[0]
}).sort_values(by='shap_value', ascending=False)

print("\nRanking de impacto salarial por skill presente:")
print(impactos[impactos['presente'] == 1].head(10))

print("\nRecomendación formativa personalizada:")
for skill in brechas:
    print(f"- Aprende '{skill}': correlación salarial positiva observada en vacantes top.")
  ```
- **generar_estructura_extendida.py** (2.9KB)
  ```
#!/usr/bin/env python3
"""
generate_project_report.py

Genera un informe en Markdown de la estructura y contenido del proyecto,
respetando un límite práctico de tamaño de informe.
"""

import os
from pathlib import Path

# Configuración
ROOT = Path(__file__).parent.resolve()
OUTPUT = ROOT / "estructura_resumida.md"
MAX_CONTENT_BYTES = 10 * 1024      # Incluir contenido completo hasta 10 KB
MAX_PREVIEW_LINES = 50             # Líneas a mostrar si el archivo supera MAX_CONTENT_BYTES
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv"}  # directorios a ignorar

def human_size(n):
    """Convierte bytes a una cadena más legible."""
    for unit in ("B","KB","MB","GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def is_text_file(path: Path):
    """Prueba rápida: ext de texto común."""
    return path.suffix.lower() in {
        ".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".ini", ".cfg", ".xml", ".html"
    }

def main():
    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("# Informe de la estructura del proyecto\n\n")
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # Saltar entornos y .git
            rel_dir = Path(dirpath).relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel_dir.parts):
                continue

            # Escribir cabecera de directorio
            depth = len(rel_dir.parts)
            out.write(f"{'#'*(depth+1)} {rel_dir or 'Raíz'}\n\n")

            for fn in sorted(filenames):
                path = Path(dirpath) / fn
                # Omitir el propio informe
                if path == OUTPUT:
                    continue

                size = path.stat().st_size
                out.write(f"- **{rel_dir / fn}** ({human_size(size)})\n")

                # Si es archivo de texto pequeño, incluir completo
                if is_text_file(path) and size <= MAX_CONTENT_BYTES:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    out.write("  ```\n")
                    out.write(text.rstrip() + "\n")
                    out.write("  ```\n")
                # Si es texto grande, incluir sólo un preview
                elif is_text_file(path):
                    out.write("  ```\n")
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= MAX_PREVIEW_LINES:
                                out.write(f"... _solo primeras {MAX_PREVIEW_LINES} líneas_\n")
                                break
                            out.write(line.rstrip() + "\n")
                    out.write("  ```\n")

                # Para binarios o extensiones no listadas, no muestro contenido
            out.write("\n")

    print(f"✔ Informe generado: {OUTPUT}")

if __name__ == "__main__":
    main()
  ```
- **generar_informe.py** (2.7KB)
  ```
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import pandas as pd
import os

# ========= CONFIGURACIÓN =========
OUTPUT_PDF = "reports/informe_modelado_ml.pdf"
STRUCTURED_FILE = "data/ofertas_variables_semanticas.csv"
RAW_FILE = "data/raw/sample_jobs.csv"
IMG_PATH = "reports/figures/feature_importance.png"

def parse_salary(s):
    if pd.isna(s): return None
    s = str(s).replace("$", "").replace(",", "").strip()
    if "-" in s:
        try:
            low, high = s.split("-")
            return (float(low) + float(high)) / 2
        except:
            return None
    try:
        return float(s)
    except:
        return None

def generar_informe():
    # =====================
    # CARGA Y PREPROCESO
    # =====================
    df = pd.read_csv(STRUCTURED_FILE)
    df_raw = pd.read_csv(RAW_FILE)

    df["salary"] = df_raw["salary"]
    df["salary_numeric"] = df["salary"].apply(parse_salary)

    total = len(df)
    media = round(df["salary_numeric"].mean(), 2)
    minimo = round(df["salary_numeric"].min(), 2)
    maximo = round(df["salary_numeric"].max(), 2)
    variables = len([col for col in df.columns if col.startswith(("skill_", "cert_", "edu_"))])

    # =====================
    # CREACIÓN DEL PDF
    # =====================
    c = canvas.Canvas(OUTPUT_PDF, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2.5*cm, h - 2.5*cm, "📊 Informe Final: Modelado ML y Variables Salariales")

    c.setFont("Helvetica", 11)
    c.drawString(2.5*cm, h - 3.5*cm, "Este informe resume los resultados del modelado predictivo sobre salario.")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.5*cm, h - 5*cm, "📌 Resumen de Datos:")

    c.setFont("Helvetica", 10)
    c.drawString(3*cm, h - 6*cm, f"Total de ofertas analizadas: {total}")
    c.drawString(3*cm, h - 6.6*cm, f"Salario promedio: ${media}")
    c.drawString(3*cm, h - 7.2*cm, f"Rango salarial: ${minimo} – ${maximo}")
    c.drawString(3*cm, h - 7.8*cm, f"Número de variables predictoras: {variables}")

    if os.path.exists(IMG_PATH):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2.5*cm, h - 9.5*cm, "🎯 Importancia de Variables según XGBoost + SHAP")
        c.drawImage(IMG_PATH, 3*cm, h - 21*cm, width=12*cm, preserveAspectRatio=True)

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2.5*cm, 2*cm, "Generado automáticamente con ReportLab – Proyecto Analítica Laboral 2025")

    c.showPage()
    c.save()
    print(f"✅ Informe PDF generado correctamente: {OUTPUT_PDF}")

if __name__ == "__main__":
    generar_informe()
  ```
- **generar_informe_data_analyst.py** (5.4KB)
  ```
# generar_informe_data_analyst.py
# Script definitivo para generar informe completo, detallado y profesional sobre formación y salario en Data Analyst

import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from collections import Counter
import spacy
import os

# ----------- CONFIGURACIÓN -----------
CSV_FILE = "sample_jobs.csv"
OUTPUT_PDF = "informe_formacion_data_analyst_completo.pdf"
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

# Keywords detallados
BACHELOR_TERMS = ["bachelor of science", "bachelor’s degree", "b.sc", "degree in statistics"]
MASTER_TERMS = ["master of science", "master’s in data science", "msc data"]
PHD_TERMS = ["phd", "doctorate"]
CERTIFICATIONS = ["google data analytics", "ibm data analyst", "excel", "tableau", "power bi", "azure"]

nlp = spacy.load("en_core_web_sm")

# Función para limpiar texto incompatible con FPDF
def safe_text(text):
    return (text.replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
                .replace("–", "-")
                .replace("…", "...")
                .replace("→", "->")
                .replace("–", "-"))

# ----------- FUNCIONES -----------
def cargar_datos():
    df = pd.read_csv(CSV_FILE)
    df.dropna(subset=["description"], inplace=True)
    return df

def extraer_info(descripciones):
    b_count, m_count, p_count, c_count = Counter(), Counter(), Counter(), Counter()
    for desc in descripciones:
        doc = nlp(desc.lower())
        for sent in doc.sents:
            s = sent.text
            for b in BACHELOR_TERMS:
                if b in s:
                    b_count[b] += 1
            for m in MASTER_TERMS:
                if m in s:
                    m_count[m] += 1
            for p in PHD_TERMS:
                if p in s:
                    p_count[p] += 1
            for c in CERTIFICATIONS:
                if c in s:
                    c_count[c] += 1
    return b_count, m_count, p_count, c_count

def parse_salary(s):
    import re
    if pd.isna(s) or "$" not in s:
        return None
    nums = re.findall(r"\$([0-9,]+)", s)
    try:
        nums = [int(n.replace(",", "")) for n in nums]
        return sum(nums)/len(nums) if nums else None
    except:
        return None

def asociar_salario(df, categorias):
    df["parsed_salary"] = df["salary"].apply(parse_salary)
    resultados = []
    for _, row in df.iterrows():
        desc = row["description"].lower()
        for grupo, claves in categorias.items():
            for clave in claves:
                if clave in desc:
                    resultados.append((grupo, row["parsed_salary"]))
    df_sal = pd.DataFrame(resultados, columns=["Formacion", "Salary"])
    return df_sal.dropna()

def graficar(datos, titulo, nombre):
    df = pd.DataFrame(datos.items(), columns=["Etiqueta", "Frecuencia"])
    df = df.sort_values(by="Frecuencia", ascending=False)
    path = f"{IMG_DIR}/{nombre}.png"
    plt.figure(figsize=(8,4))
    plt.bar(df["Etiqueta"], df["Frecuencia"])
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# ----------- INFORME PDF -----------
def crear_pdf(b, m, p, c, img_b, img_m, img_p, img_c, img_sal, sal_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, safe_text("Informe Profesional: Formacion y Salario (Data Analyst)"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, safe_text(
        "Este informe presenta un analisis preciso y completo de las formaciones y certificaciones mas mencionadas en \
        ofertas de Data Analyst, asi como su relacion con el salario promedio."
    ))

    def seccion(titulo, datos, img):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, safe_text(titulo), ln=True)
        pdf.set_font("Arial", '', 11)
        for k, v in datos.items():
            pdf.cell(0, 8, safe_text(f" - {k}: {v}"), ln=True)
        pdf.image(img, w=180)

    seccion("1. Grados universitarios", b, img_b)
    seccion("2. Masters", m, img_m)
    seccion("3. Doctorados", p, img_p)
    seccion("4. Certificaciones", c, img_c)

    # Salario
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, safe_text("5. Salario promedio por formacion"), ln=True)
    resumen = sal_df.groupby("Formacion")["Salary"].mean().sort_values(ascending=False)
    pdf.set_font("Arial", '', 11)
    for k, v in resumen.items():
        pdf.cell(0, 8, safe_text(f" - {k}: ${v:,.2f}"), ln=True)
    pdf.image(img_sal, w=180)

    # Guardar
    pdf.output(OUTPUT_PDF)

# ----------- EJECUCIÓN -----------
if __name__ == "__main__":
    df = cargar_datos()
    b, m, p, c = extraer_info(df["description"].tolist())
    img_b = graficar(b, "Grados", "bachelors")
    img_m = graficar(m, "Masters", "masters")
    img_p = graficar(p, "Doctorados", "phds")
    img_c = graficar(c, "Certificaciones", "certs")
    sal = asociar_salario(df, {
        "Bachelor": list(b.keys()),
        "Master": list(m.keys()),
        "PhD": list(p.keys()),
        "Certificacion": list(c.keys())
    })
    img_sal = graficar(sal.groupby("Formacion")["Salary"].mean().to_dict(), "Salario promedio", "salarios")
    crear_pdf(b, m, p, c, img_b, img_m, img_p, img_c, img_sal, sal)
    print("✅ Informe generado con éxito:", OUTPUT_PDF)
  ```
- **generate_correlation_heatmap.py** (1.3KB)
  ```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main(input_csv, output_image):
    print("📥 Leyendo archivo...")
    df = pd.read_csv(input_csv)

    # Filtrar solo columnas numéricas
    numeric_df = df.select_dtypes(include='number')

    if 'salary_numeric' not in numeric_df.columns:
        raise ValueError("❌ La columna 'salary_numeric' no está presente en el archivo.")

    print("📊 Calculando matriz de correlación...")
    corr = numeric_df.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Heatmap de Correlaciones", fontsize=16)
    plt.tight_layout()

    output_dir = os.path.dirname(output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_image)
    print(f"✅ Heatmap guardado como: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--output", default="reports/heatmap_correlaciones.png", help="Ruta del archivo PNG de salida")
    args = parser.parse_args()

    main(args.input, args.output)
  ```
- **generate_project_report.py** (2.9KB)
  ```
#!/usr/bin/env python3
"""
generate_project_report.py

Genera un informe en Markdown de la estructura y contenido del proyecto,
respetando un límite práctico de tamaño de informe.
"""

import os
from pathlib import Path

# Configuración
ROOT = Path(__file__).parent.resolve()
OUTPUT = ROOT / "estructura_resumida.md"
MAX_CONTENT_BYTES = 10 * 1024      # Incluir contenido completo hasta 10 KB
MAX_PREVIEW_LINES = 50             # Líneas a mostrar si el archivo supera MAX_CONTENT_BYTES
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv"}  # directorios a ignorar

def human_size(n):
    """Convierte bytes a una cadena más legible."""
    for unit in ("B","KB","MB","GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"

def is_text_file(path: Path):
    """Prueba rápida: ext de texto común."""
    return path.suffix.lower() in {
        ".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".ini", ".cfg", ".xml", ".html"
    }

def main():
    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write("# Informe de la estructura del proyecto\n\n")
        for dirpath, dirnames, filenames in os.walk(ROOT):
            # Saltar entornos y .git
            rel_dir = Path(dirpath).relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel_dir.parts):
                continue

            # Escribir cabecera de directorio
            depth = len(rel_dir.parts)
            out.write(f"{'#'*(depth+1)} {rel_dir or 'Raíz'}\n\n")

            for fn in sorted(filenames):
                path = Path(dirpath) / fn
                # Omitir el propio informe
                if path == OUTPUT:
                    continue

                size = path.stat().st_size
                out.write(f"- **{rel_dir / fn}** ({human_size(size)})\n")

                # Si es archivo de texto pequeño, incluir completo
                if is_text_file(path) and size <= MAX_CONTENT_BYTES:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    out.write("  ```\n")
                    out.write(text.rstrip() + "\n")
                    out.write("  ```\n")
                # Si es texto grande, incluir sólo un preview
                elif is_text_file(path):
                    out.write("  ```\n")
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= MAX_PREVIEW_LINES:
                                out.write(f"... _solo primeras {MAX_PREVIEW_LINES} líneas_\n")
                                break
                            out.write(line.rstrip() + "\n")
                    out.write("  ```\n")

                # Para binarios o extensiones no listadas, no muestro contenido
            out.write("\n")

    print(f"✔ Informe generado: {OUTPUT}")

if __name__ == "__main__":
    main()
  ```
- **generate_structured_dataset.py** (3.5KB)
  ```
# scripts/generate_structured_dataset.py

import pandas as pd
import numpy as np
import argparse
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_salary(salary_str):
    """
    Convierte rangos salariales en string a un número promedio anual.
    """
    if not isinstance(salary_str, str):
        return np.nan
    # Reemplaza 'k' por '000' y extrae números
    cleaned = salary_str.lower().replace('k', '000')
    nums = re.findall(r"\d+\.?\d*", cleaned)
    if not nums:
        return np.nan
    vals = list(map(float, nums))
    return np.mean(vals)

def generate_features(df, salary_col, normalize_flag, tfidf_flag, tfidf_n):
    # 1) Salario
    if normalize_flag:
        df['salary_normalized'] = df[salary_col].apply(normalize_salary)
    else:
        df['salary_normalized'] = df[salary_col]

    # 2) Columnas semánticas binarias
    prefixes = ['skill_', 'tool_', 'certification_', 'education_', 'language_', 'seniority_']
    entity_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    entity_df = df[entity_cols].fillna(0).astype(int)

    # 3) TF-IDF de descripciones
    if tfidf_flag:
        vect = TfidfVectorizer(max_features=tfidf_n, stop_words='english')
        tfidf_mat = vect.fit_transform(df['description'].astype(str))
        tfidf_df = pd.DataFrame(
            tfidf_mat.toarray(),
            columns=[f"tfidf_{w}" for w in vect.get_feature_names_out()],
            index=df.index
        )
    else:
        tfidf_df = pd.DataFrame(index=df.index)

    # 4) Otras variables
    if 'description' in df.columns:
        df['description_length'] = df['description'].astype(str).str.len()
    else:
        df['description_length'] = 0

        df['num_entities'] = entity_df.sum(axis=1)

    # 5) Concatenar todas las features
    features = pd.concat([
        df[['salary_normalized', 'description_length', 'num_entities']],
        entity_df,
        tfidf_df
    ], axis=1)

    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera dataset de features con TF-IDF opcional'
    )
    parser.add_argument(
        '--input', required=True,
        help='CSV semántico de entrada'
    )
    parser.add_argument(
        '--output', required=True,
        help='CSV de salida de features'
    )
    parser.add_argument(
        '--salary-column', dest='sal_col', default='salary',
        help='Columna salarial original'
    )
    parser.add_argument(
        '--normalize-salary', dest='normalize_flag',
        action='store_true',
        help='Normaliza salario'
    )
    parser.add_argument(
        '--tfidf', dest='tfidf_flag',
        action='store_true',
        help='Incluir TF-IDF'
    )
    parser.add_argument(
        '--tfidf-features', dest='tfidf_n', type=int,
        default=50,
        help='Número de términos TF-IDF'
    )
    args = parser.parse_args()

    # Cargar semántico
    print(f"📥 Cargando datos desde: {args.input}")
    df = pd.read_csv(args.input)

    # Generar features
    print("⚙️ Generando features...")
    features_df = generate_features(
        df,
        salary_col=args.sal_col,
        normalize_flag=args.normalize_flag,
        tfidf_flag=args.tfidf_flag,
        tfidf_n=args.tfidf_n
    )

    # Guardar
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"✅ Features guardadas en: {args.output}")
  ```
- **generate_test_offers.py** (1004.0B)
  ```
import pandas as pd

# Columnas esperadas por el modelo, incluyendo description_id
data = {
    "description_id": [1, 2, 3],
    "certification_google_data_analytics": [1, 0, 1],
    "other_au": [0, 1, 0],
    "other_azure": [0, 0, 1],
    "other_b": [0, 1, 0],
    "other_business": [1, 1, 0],
    "other_computer_science": [0, 0, 1],
    "other_data_science": [1, 1, 1],
    "other_i": [0, 0, 0],
    "other_ibm_data_science_certificate": [1, 0, 0],
    "other_informatics": [0, 1, 0],
    "other_marketing": [0, 0, 0],
    "other_power": [0, 1, 1],
    "other_strong": [0, 0, 0],
    "other_table": [0, 0, 1],
    "skill_excel": [1, 1, 1],
    "skill_power_bi": [1, 0, 1],
    "skill_python": [1, 1, 1],
    "skill_r": [0, 1, 0],
    "skill_sql": [1, 1, 1],
    "skill_statistics": [1, 0, 1],
    "skill_tableau": [0, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv("data/nuevas_ofertas.csv", index=False)
print("✅ Archivo generado correctamente: data/nuevas_ofertas.csv")
  ```
- **launcher.py** (1.0KB)
  ```
import subprocess
import os
import time

def ejecutar(nombre_script):
    print(f"\n🚀 Ejecutando: {nombre_script}")
    resultado = subprocess.run(["python", nombre_script], capture_output=True, text=True)
    print(resultado.stdout)
    if resultado.stderr:
        print("⚠️ Error:", resultado.stderr)

def main():
    # 1. Pipeline completo de preprocesamiento
    ejecutar("scripts/main_pipeline_jobs.py")

    # 2. Modelado predictivo y SHAP
    ejecutar("scripts/analysis_and_modeling_advanced.py")

    # 3. Informe estructurado ML
    ejecutar("scripts/generar_informe.py")

    # 4. Informe semántico textual con spaCy
    ejecutar("scripts/generar_informe_data_analyst.py")

    print("\n✅ Sistema completo ejecutado correctamente.")

    # 5. Opción para abrir el dashboard
    respuesta = input("\n¿Deseas abrir el dashboard interactivo ahora? (s/n): ").strip().lower()
    if respuesta == "s":
        subprocess.run(["streamlit", "run", "scripts/dashboard.py"])

if __name__ == "__main__":
    main()
  ```
- **main_pipeline_jobs.py** (6.1KB)
  ```
# scripts/main_pipeline_jobs.py

import os
import pandas as pd
from tqdm import tqdm
import re

# Preprocesamiento
from langdetect import detect
from deep_translator import GoogleTranslator

# Transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ====================
# CONFIGURACIÓN GENERAL
# ====================
INPUT_FILE = "data/raw/sample_jobs.csv"
PREPROCESSED_FILE = "data/job_postings_preprocessed.csv"
NER_OUTPUT_FILE = "data/entities_extracted.csv"
FINAL_OUTPUT_FILE = "data/final_entities_semantic.csv"
STRUCTURED_OUTPUT_FILE = "data/ofertas_variables_semanticas.csv"
TEXT_COLUMN = "description"

MODEL_NAME = "dslim/bert-base-NER"
os.makedirs("data", exist_ok=True)

# ====================
# FUNCIONES AUXILIARES
# ====================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text):
    lang = detect_language(text)
    if lang != 'en' and lang != 'unknown':
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text
    return text

def preprocess_job_posting(text):
    text = clean_text(text)
    text = translate_to_english(text)
    return text

def load_ner_pipeline(model_name=MODEL_NAME):
    print(f"🔄 Cargando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(df: pd.DataFrame, text_column: str, ner_pipeline, output_path: str):
    print("🧠 Extrayendo entidades con NER Transformer...")
    all_entities = []
    for i, text in tqdm(enumerate(df[text_column].fillna("")), total=len(df)):
        entities = ner_pipeline(text)
        for ent in entities:
            all_entities.append({
                "description_id": i,
                "entity": ent["word"],
                "entity_group": ent["entity_group"],
                "score": round(ent["score"], 4),
                "start": ent["start"],
                "end": ent["end"]
            })
    df_entities = pd.DataFrame(all_entities)
    df_entities.to_csv(output_path, index=False)
    print(f"✅ Entidades guardadas en: {output_path}")
    return df_entities

# ====================
# CLASIFICACIÓN SEMÁNTICA
# ====================
SKILLS = {'python', 'sql', 'r', 'power bi', 'excel', 'tableau', 'machine learning', 'deep learning', 'statistics'}
TOOLS = {'tensorflow', 'pytorch', 'keras', 'airflow', 'docker', 'spark', 'git'}
CERTIFICATIONS = {'google data analytics', 'aws certified', 'azure fundamentals', 'data science specialization'}
EDUCATION = {'bachelor', 'master', 'phd', 'licenciatura', 'grado', 'máster', 'doctorado'}
LANGUAGES = {'english', 'spanish', 'german', 'french'}
SENIORITY = {'junior', 'mid-level', 'senior', 'lead', 'principal'}

def classify_entity(entity: str) -> str:
    ent = entity.lower().strip()
    if any(re.search(rf'\b{word}\b', ent) for word in CERTIFICATIONS):
        return 'Certification'
    if any(re.search(rf'\b{word}\b', ent) for word in EDUCATION):
        return 'Education'
    if any(re.search(rf'\b{word}\b', ent) for word in SKILLS):
        return 'Skill'
    if any(re.search(rf'\b{word}\b', ent) for word in TOOLS):
        return 'Tool'
    if any(re.search(rf'\b{word}\b', ent) for word in LANGUAGES):
        return 'Language'
    if any(re.search(rf'\b{word}\b', ent) for word in SENIORITY):
        return 'Seniority'
    return 'Other'

def classify_entities(df: pd.DataFrame) -> pd.DataFrame:
    print("🔍 Clasificando entidades semánticamente...")
    df['semantic_class'] = df['entity'].apply(classify_entity)
    return df

# ====================
# GENERACIÓN DEL DATASET ESTRUCTURADO
# ====================
def generate_structured_dataset(df_classified: pd.DataFrame, original_df: pd.DataFrame, output_path: str):
    print("🔄 Generando variables binarias por oferta...")
    df = df_classified.copy()
    df["entity_clean"] = (
        df["entity"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    df["feature"] = df["semantic_class"].str.lower() + "_" + df["entity_clean"]
    df_unique = df.drop_duplicates(subset=["description_id", "feature"])
    df_features = (
        df_unique[["description_id", "feature"]]
        .assign(value=1)
        .pivot_table(index="description_id", columns="feature", values="value", fill_value=0)
        .reset_index()
    )

    # 🔁 Añadir columnas originales útiles
    for col in ["salary", "title", "description"]:
        if col in original_df.columns:
            df_features[col] = original_df[col]

    df_features.to_csv(output_path, index=False)
    print(f"✅ Dataset estructurado guardado como: {output_path}")

# ====================
# FLUJO PRINCIPAL
# ====================
if __name__ == "__main__":
    print(f"📥 Leyendo datos desde: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print("🔧 Preprocesando descripciones...")
    tqdm.pandas(desc="📚 NLP")
    df["description_clean"] = df[TEXT_COLUMN].progress_apply(preprocess_job_posting)
    df.to_csv(PREPROCESSED_FILE, index=False)
    print(f"✅ Archivo preprocesado guardado en: {PREPROCESSED_FILE}")

    ner = load_ner_pipeline()
    df_entities = extract_entities(df, "description_clean", ner, NER_OUTPUT_FILE)

    df_classified = classify_entities(df_entities)
    df_classified.to_csv(FINAL_OUTPUT_FILE, index=False)
    print(f"✅ Clasificación final guardada en: {FINAL_OUTPUT_FILE}")

    generate_structured_dataset(df_classified, df, STRUCTURED_OUTPUT_FILE)
  ```
- **metrics.json** (565.0B)
  ```
{
  "Linear": {
    "mae": 7.033425693710645e-11,
    "mse": 8.364450354136216e-21,
    "r2": 1.0
  },
  "Poly2": {
    "mae": 1.2126596023639044e-11,
    "mse": 3.8822343415822097e-22,
    "r2": 1.0
  },
  "LightGBM": {
    "mae": 11666.666666666666,
    "mse": 193055555.55555555,
    "r2": 0.16666666666666663
  },
  "CatBoost": {
    "mae": 114.20819711232132,
    "mse": 19515.632220258336,
    "r2": 0.9999157598609197
  },
  "Stack": {
    "mae": 17766.227839523155,
    "mse": 495995423.5588973,
    "r2": -1.1409874398225783
  }
}
  ```
- **metrics_train.json** (565.0B)
  ```
{
  "Linear": {
    "mae": 7.033425693710645e-11,
    "mse": 8.364450354136216e-21,
    "r2": 1.0
  },
  "Poly2": {
    "mae": 1.2126596023639044e-11,
    "mse": 3.8822343415822097e-22,
    "r2": 1.0
  },
  "LightGBM": {
    "mae": 11666.666666666666,
    "mse": 193055555.55555555,
    "r2": 0.16666666666666663
  },
  "CatBoost": {
    "mae": 114.20819711232132,
    "mse": 19515.632220258336,
    "r2": 0.9999157598609197
  },
  "Stack": {
    "mae": 17722.725281557872,
    "mse": 493810353.6145875,
    "r2": -1.1315554832284351
  }
}
  ```
- **ner.py** (4.0KB)
  ```
# scripts/ner.py

import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
)

# 1. Definición de etiquetas y mapeos
LABEL_LIST = [
    "O",
    "B-Skill", "I-Skill",
    "B-Tool", "I-Tool",
    "B-Certification", "I-Certification",
    "B-Education", "I-Education",
    "B-Language", "I-Language",
    "B-Seniority", "I-Seniority"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

# 2. Carga del dataset anotado
dataset = load_dataset(
    "json",
    data_files={"train": "data/annotated/ner_dataset.jsonl"}
)

# 3. División en train/test (90% / 10%)
split = dataset["train"].train_test_split(test_size=0.1, seed=42)
split["validation"] = split.pop("test")

# 4. Preparar tokenizer y modelo
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# 5. Función de tokenización y alineación de etiquetas
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length"
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[label[word_id]])
            else:
                current_label = label[word_id]
                if current_label.startswith("B-"):
                    aligned_labels.append(LABEL2ID[current_label.replace("B-", "I-")])
                else:
                    aligned_labels.append(LABEL2ID[current_label])
            prev_word_id = word_id
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 6. Aplicar tokenización al dataset
tokenized_datasets = {
    split_name: ds.map(tokenize_and_align_labels, batched=True)
    for split_name, ds in split.items()
}

# 7. Métrica de evaluación
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

# 8. Configuración de entrenamiento (sin evaluation_strategy)
training_args = TrainingArguments(
    output_dir="models/ner_model",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    eval_steps=50,
    save_steps=50,
    save_total_limit=2
)

# 9. Entrenamiento y guardado
def main():
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model("models/ner_model")
    tokenizer.save_pretrained("models/ner_model")
    print("✅ Fine-tuning completado y modelo guardado en models/ner_model")

if __name__ == "__main__":
    main()
  ```
- **ner_transformers_extraction.py** (1.7KB)
  ```
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import os

# Configuración general
MODEL_NAME = "dslim/bert-base-NER"
INPUT_FILE = "data/job_postings_preprocessed.csv"  # archivo con las descripciones preprocesadas
TEXT_COLUMN = "description_clean"  # columna con texto limpio
OUTPUT_FILE = "output/ner_extracted_entities.csv"
os.makedirs("output", exist_ok=True)

# Carga del modelo y pipeline
print(f"Cargando modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Carga de datos
print(f"Leyendo datos desde: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"La columna '{TEXT_COLUMN}' no se encuentra en el archivo de entrada.")

# Extracción de entidades
all_entities = []

print("Extrayendo entidades con NER Transformer...")
for i, text in tqdm(enumerate(df[TEXT_COLUMN].fillna("")), total=len(df)):
    entities = ner_pipeline(text)
    for ent in entities:
        all_entities.append({
            "post_index": i,
            "entity": ent["word"],
            "entity_group": ent["entity_group"],
            "score": round(ent["score"], 4),
            "start": ent["start"],
            "end": ent["end"]
        })

# Creación de DataFrame con resultados
df_entities = pd.DataFrame(all_entities)
df_entities.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Entidades extraídas y guardadas en: {OUTPUT_FILE}")
print("Ejemplo de entidades extraídas:")
print(df_entities.head(10))
  ```
- **onenote_extractor.py** (3.6KB)
  ```
import os
import json
import requests
import logging
from msal import PublicClientApplication
from bs4 import BeautifulSoup
from datetime import datetime

# Configura los logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# -------------------- CONFIGURACIÓN --------------------
CLIENT_ID = 'TU_CLIENT_ID'
TENANT_ID = 'TU_TENANT_ID'
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ['Notes.Read.All']
OUTPUT_DIR = "OneNote_Export"
# -------------------------------------------------------

# Inicializa MSAL para autenticación
app = PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY)
result = app.acquire_token_interactive(scopes=SCOPES)
token = result.get("access_token")

headers = {
    'Authorization': f'Bearer {token}'
}

# Crea directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Función para limpiar contenido HTML
def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator='\n').strip()

# Función principal de extracción
def extract_onenote():
    all_data = []

    logging.info("Obteniendo blocs de notas...")
    notebooks = requests.get('https://graph.microsoft.com/v1.0/me/onenote/notebooks', headers=headers).json().get('value', [])

    for nb in notebooks:
        nb_name = nb['displayName'].strip().replace("/", "_")
        logging.info(f"📒 Bloc: {nb_name}")
        nb_path = os.path.join(OUTPUT_DIR, nb_name)
        os.makedirs(nb_path, exist_ok=True)

        sections = requests.get(nb['sectionsUrl'], headers=headers).json().get('value', [])
        for sec in sections:
            sec_name = sec['displayName'].strip().replace("/", "_")
            logging.info(f"  📂 Sección: {sec_name}")
            sec_path = os.path.join(nb_path, sec_name)
            os.makedirs(sec_path, exist_ok=True)

            pages_url = f"https://graph.microsoft.com/v1.0/me/onenote/sections/{sec['id']}/pages"
            pages = requests.get(pages_url, headers=headers).json().get('value', [])

            for pg in pages:
                title = pg['title'].strip().replace("/", "_")
                page_id = pg['id']
                page_content_url = pg['contentUrl']
                logging.info(f"    📄 Página: {title}")

                try:
                    html_resp = requests.get(page_content_url, headers=headers)
                    text_content = extract_text_from_html(html_resp.text)

                    filename = os.path.join(sec_path, f"{title}.txt")
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(text_content)

                    all_data.append({
                        "notebook": nb_name,
                        "section": sec_name,
                        "page": title,
                        "content": text_content
                    })

                except Exception as e:
                    logging.warning(f"❌ Error procesando {title}: {e}")
    
    # Exporta CSV y JSON
    export_json = os.path.join(OUTPUT_DIR, "export_onenote.json")
    export_csv = os.path.join(OUTPUT_DIR, "export_onenote.csv")

    with open(export_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    import pandas as pd
    df = pd.DataFrame(all_data)
    df.to_csv(export_csv, index=False, encoding="utf-8")

    logging.info(f"✅ Exportación finalizada: {len(all_data)} páginas procesadas.")
    logging.info(f"📁 Archivos guardados en: {OUTPUT_DIR}")

# Ejecuta todo
if __name__ == "__main__":
    extract_onenote()
  ```
- **predict_salary.py** (1.3KB)
  ```
import argparse
import pandas as pd
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Ruta del CSV con nuevas ofertas")
parser.add_argument("--model", required=True, help="Ruta al modelo entrenado .pkl")
parser.add_argument("--output", required=True, help="Ruta de salida del CSV con predicciones")
args = parser.parse_args()

print("📥 Leyendo archivo de entrada...")
df = pd.read_csv(args.input)

print("🧠 Cargando modelo entrenado...")
model = joblib.load(args.model)

# Asegurar que las columnas estén en el mismo orden y conjunto que en el entrenamiento
if hasattr(model, "feature_names_in_"):
    expected_features = model.feature_names_in_
else:
    expected_features = model.estimators_[0][1].feature_names_in_

missing = [col for col in expected_features if col not in df.columns]
extra = [col for col in df.columns if col not in expected_features]

if missing:
    raise ValueError(f"❌ Faltan columnas requeridas por el modelo: {missing}")
if extra:
    print(f"⚠️ Columnas extra ignoradas: {extra}")

X = df[expected_features]

print("🔮 Realizando predicciones de salario...")
predicciones = model.predict(X)
df["predicted_salary"] = predicciones

df.to_csv(args.output, index=False)
print(f"✅ Predicciones guardadas en: {args.output}")
  ```
- **preprocess_csv.py** (764.0B)
  ```
import pandas as pd
from tqdm import tqdm
from preprocessing_nlp import preprocess_job_posting  # Usa tu script existente

INPUT_FILE = "sample_jobs.csv"  # Asegúrate que existe
OUTPUT_FILE = "data/job_postings_preprocessed.csv"
TEXT_COLUMN = "description"  # Cambia si tu columna se llama distinto

# Leer archivo
df = pd.read_csv(INPUT_FILE)
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"La columna '{TEXT_COLUMN}' no existe en el archivo.")

# Preprocesar cada descripción
tqdm.pandas(desc="Preprocesando descripciones")
df["description_clean"] = df[TEXT_COLUMN].progress_apply(preprocess_job_posting)

# Guardar archivo limpio
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Archivo preprocesado guardado en: {OUTPUT_FILE}")
  ```
- **preprocessing.py** (5.4KB)
  ```
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
  ```
- **preprocessing_all.py** (958.0B)
  ```
# preprocessing_all.py

import pandas as pd
from tqdm import tqdm
from preprocessing_nlp import preprocess_job_posting  # Asegúrate de tener este script en el mismo directorio o en tu PYTHONPATH

INPUT_FILE = "sample_jobs.csv"
OUTPUT_FILE = "data/job_postings_preprocessed.csv"
TEXT_COLUMN = "description"  # Cambia este nombre si tu CSV usa otra columna para la descripción

# Crear carpeta de salida si no existe
import os
os.makedirs("data", exist_ok=True)

# Leer el archivo de entrada
df = pd.read_csv(INPUT_FILE)
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"La columna '{TEXT_COLUMN}' no existe en el archivo {INPUT_FILE}")

# Aplicar el preprocesamiento a cada texto
tqdm.pandas(desc="Preprocesando textos")
df["description_clean"] = df[TEXT_COLUMN].progress_apply(preprocess_job_posting)

# Guardar resultados
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Archivo preprocesado guardado en: {OUTPUT_FILE}")
  ```
- **preprocessing_nlp.py** (1.4KB)
  ```
import re
from langdetect import detect
from deep_translator import GoogleTranslator


def clean_text(text):
    """Limpieza general del texto."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)  # eliminar HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)  # eliminar URLs
    text = re.sub(r'[^\w\s.,]', '', text)  # eliminar símbolos
    text = re.sub(r'\s+', ' ', text)  # espacios múltiples
    return text.strip()


def detect_language(text):
    """Detecta el idioma del texto."""
    try:
        return detect(text)
    except:
        return "unknown"


def translate_to_english(text):
    """Traduce el texto al inglés si no está en inglés."""
    lang = detect_language(text)
    if lang != 'en' and lang != 'unknown':
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        except:
            return text
    return text


def preprocess_job_posting(text):
    """Pipeline de preprocesamiento completo para una oferta de empleo."""
    text = clean_text(text)
    text = translate_to_english(text)
    return text


# Ejemplo de uso
if __name__ == "__main__":
    raw_text = "<div>Se busca Data Analyst con experiencia en Python, SQL y Power BI. Salario: 50.000€</div>"
    processed = preprocess_job_posting(raw_text)
    print(processed)
  ```
- **prueba.py** (687.0B)
  ```
import pandas as pd

# Cargar ambos archivos completos
df_ofertas = pd.read_csv('data/nuevas_ofertas.csv')
df_salarios = pd.read_csv('data/predicciones_salariales.csv')

# Verificamos que tenga la columna 'description'
assert 'description' in df_ofertas.columns, "❌ Falta columna 'description' en nuevas_ofertas.csv"

# Hacer merge conservando la descripción
df_merged = df_ofertas.merge(df_salarios[['description_id', 'predicted_salary']], on='description_id', how='left')

# Guardar con descripción incluida
df_merged.to_csv('data/nuevas_ofertas_con_salario.csv', index=False)
print("✅ Archivo actualizado con descripción: 'data/nuevas_ofertas_con_salario.csv'")
  ```
- **semantic_classifier.py** (3.0KB)
  ```
# scripts/semantic_classifier.py

import pandas as pd
import argparse
import os
import ast

def classify_list(entity_strings):
    """
    Convierte una lista de strings "Etiqueta:Texto" en un dict de columnas semánticas binarias.
    """
    semantic = {}
    # Si la columna viene como string, parsearla a lista
    if isinstance(entity_strings, str):
        try:
            entity_strings = ast.literal_eval(entity_strings)
        except Exception:
            return semantic
    if not isinstance(entity_strings, list):
        return semantic

    for ent in entity_strings:
        if not ent or ':' not in ent:
            continue
        label, word = ent.split(':', 1)
        label_low = label.strip().lower()
        word_key = word.strip().lower().replace(' ', '_')
        key = None
        if label_low == 'skill':
            key = f"skill_{word_key}"
        elif label_low == 'tool':
            key = f"tool_{word_key}"
        elif label_low == 'certification':
            key = f"certification_{word_key}"
        elif label_low == 'education':
            key = f"education_{word_key}"
        elif label_low == 'language':
            key = f"language_{word_key}"
        elif label_low == 'seniority':
            key = f"seniority_{word_key}"
        if key:
            semantic[key] = 1
    return semantic

def classify_entities_df(df, col_name):
    """
    Aplica `classify_list` a cada fila y expande los dicts en columnas.
    """
    df = df.copy()
    semantic_series = df[col_name].apply(classify_list)
    semantic_df = pd.DataFrame(list(semantic_series)).fillna(0).astype(int)
    df = pd.concat([df, semantic_df], axis=1)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clasifica semánticamente entidades extraídas de ofertas de empleo'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Ruta al CSV con columna de entidades extraídas'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Ruta de salida para el CSV con columnas semánticas'
    )
    parser.add_argument(
        '--column', type=str, default='extracted_entities',
        help='Nombre de la columna con la lista de entidades'
    )
    args = parser.parse_args()

    # Leer input
    print(f"📥 Leyendo entidades desde: {args.input}")
    df = pd.read_csv(args.input)

    if args.column not in df.columns:
        raise KeyError(
            f"La columna '{args.column}' no existe en {args.input}. "
            f"Columnas disponibles: {', '.join(df.columns)}"
        )

    # Clasificar
    print("🔍 Clasificando entidades semánticamente...")
    df = classify_entities_df(df, col_name=args.column)

    # Crear directorio de output si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Guardar resultado
    df.to_csv(args.output, index=False)
    print(f"💾 Guardado dataset semántico en: {args.output}")
  ```
- **split_data.py** (2.1KB)
  ```
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
  ```
- **train_model.py** (535.0B)
  ```

#!/usr/bin/env python3
# scripts/train_model.py: entrena modelo sobre datos procesados

def train(data_path, model_path):
    # TODO: cargar processed, entrenar y serializar modelo
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed', help='Carpeta processed')
    parser.add_argument('--out', default='artifacts/models', help='Dónde guardar el modelo')
    args = parser.parse_args()
    train(args.data, args.out)
  ```
- **train_salary_model.py** (2.2KB)
  ```
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURACIÓN ---
INPUT_FILE = "data/ofertas_variables_semanticas_numerico.csv"
TARGET_COLUMN = "salary_numeric"
MODEL_OUTPUT_PATH = "models/salary_predictor_final.pkl"
PLOT_OUTPUT_PATH = "feature_importance_plot.png"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# === 1. CARGAR DATOS ===
print("📥 Leyendo archivo...")
df = pd.read_csv(INPUT_FILE)

# === 2. VERIFICAR COLUMNA TARGET ===
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"❌ Error: La columna '{TARGET_COLUMN}' no existe. Columnas disponibles: {list(df.columns)}")

# === 3. PREPARAR FEATURES Y TARGET ===
X = df.drop(columns=["salary", "title", "description", TARGET_COLUMN], errors="ignore")
y = df[TARGET_COLUMN]

# === 4. DIVIDIR EN TRAIN Y TEST ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# === 5. ENTRENAR MODELO ===
print("🧠 Entrenando modelo Random Forest...")
model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# === 6. EVALUAR MODELO ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 MAE: {mae:,.2f}")
print(f"📈 R²: {r2:.4f}")

# === 7. GUARDAR MODELO ENTRENADO ===
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"✅ Modelo guardado en: {MODEL_OUTPUT_PATH}")

# === 8. IMPORTANCIA DE VARIABLES ===
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# === 9. GRAFICAR Y GUARDAR PLOT ===
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.title("🔍 Importancia de Variables en la Predicción Salarial")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig(PLOT_OUTPUT_PATH)
print(f"📸 Gráfico guardado en: {PLOT_OUTPUT_PATH}")
  ```

## artifacts


### artifacts\models

- **artifacts\models\CatBoost.joblib** (221.6KB)
- **artifacts\models\LightGBM.joblib** (10.7KB)
- **artifacts\models\Linear.joblib** (6.5KB)
- **artifacts\models\Poly2.joblib** (11.1KB)

## catboost_info

- **catboost_info\catboost_training.json** (95.4KB)
  ```
{
"meta":{"test_sets":[],"test_metrics":[],"learn_metrics":[{"best_value":"Min","name":"RMSE"}],"launch_mode":"Train","parameters":"","iteration_count":1000,"learn_sets":["learn"],"name":"experiment"},
"iterations":[
{"learn":[14879.6974],"iteration":0,"passed_time":0.1401860307,"remaining_time":140.0458447},
{"learn":[14813.75802],"iteration":1,"passed_time":0.1431153931,"remaining_time":71.41458114},
{"learn":[14737.9244],"iteration":2,"passed_time":0.1450910368,"remaining_time":48.21858788},
{"learn":[14672.61329],"iteration":3,"passed_time":0.1572710259,"remaining_time":39.16048544},
{"learn":[14612.80375],"iteration":4,"passed_time":0.1578645244,"remaining_time":31.41504035},
{"learn":[14548.4236],"iteration":5,"passed_time":0.1592484126,"remaining_time":26.38215368},
{"learn":[14483.95226],"iteration":6,"passed_time":0.1599919631,"remaining_time":22.69600277},
{"learn":[14424.9933],"iteration":7,"passed_time":0.1605406295,"remaining_time":19.90703806},
{"learn":[14381.77971],"iteration":8,"passed_time":0.1609975909,"remaining_time":17.72762362},
{"learn":[14318.37085],"iteration":9,"passed_time":0.1615127049,"remaining_time":15.98975779},
{"learn":[14251.14177],"iteration":10,"passed_time":0.1619398316,"remaining_time":14.55986304},
{"learn":[14187.98784],"iteration":11,"passed_time":0.1626671268,"remaining_time":13.39292677},
{"learn":[14130.20103],"iteration":12,"passed_time":0.1632104348,"remaining_time":12.3914384},
{"learn":[14068.03388],"iteration":13,"passed_time":0.1637438174,"remaining_time":11.53224314},
{"learn":[13998.87509],"iteration":14,"passed_time":0.1641884414,"remaining_time":10.78170765},
{"learn":[13936.83907],"iteration":15,"passed_time":0.1647542245,"remaining_time":10.1323848},
{"learn":[13865.60334],"iteration":16,"passed_time":0.1653324169,"remaining_time":9.560103873},
{"learn":[13788.37906],"iteration":17,"passed_time":0.1656550102,"remaining_time":9.037401111},
{"learn":[13727.27586],"iteration":18,"passed_time":0.1662190825,"remaining_time":8.582153682},
{"learn":[13666.44343],"iteration":19,"passed_time":0.167136563,"remaining_time":8.189691585},
{"learn":[13606.19175],"iteration":20,"passed_time":0.167657718,"remaining_time":7.81604314},
{"learn":[13545.89591],"iteration":21,"passed_time":0.1687836952,"remaining_time":7.503202452},
{"learn":[13485.86727],"iteration":22,"passed_time":0.1694421249,"remaining_time":7.197606782},
{"learn":[13426.43004],"iteration":23,"passed_time":0.1699734642,"remaining_time":6.912254211},
{"learn":[13366.93081],"iteration":24,"passed_time":0.170655757,"remaining_time":6.655574523},
{"learn":[13307.69526],"iteration":25,"passed_time":0.1714975111,"remaining_time":6.424560609},
{"learn":[13248.72221],"iteration":26,"passed_time":0.1720104918,"remaining_time":6.198748465},
{"learn":[13175.90154],"iteration":27,"passed_time":0.1725634249,"remaining_time":5.990416035},
{"learn":[13111.5145],"iteration":28,"passed_time":0.172942807,"remaining_time":5.790602263},
{"learn":[13053.41082],"iteration":29,"passed_time":0.1733543794,"remaining_time":5.605124935},
{"learn":[13000.47937],"iteration":30,"passed_time":0.1736941898,"remaining_time":5.429344192},
{"learn":[12943.09637],"iteration":31,"passed_time":0.1742626187,"remaining_time":5.271444216},
{"learn":[12885.73904],"iteration":32,"passed_time":0.1750366569,"remaining_time":5.12910446},
{"learn":[12828.8799],"iteration":33,"passed_time":0.1756220506,"remaining_time":4.989732378},
{"learn":[12772.28904],"iteration":34,"passed_time":0.1760578107,"remaining_time":4.854165353},
{"learn":[12707.17338],"iteration":35,"passed_time":0.1765706011,"remaining_time":4.728168319},
{"learn":[12637.68693],"iteration":36,"passed_time":0.1769940921,"remaining_time":4.60663002},
{"learn":[12568.6633],"iteration":37,"passed_time":0.1774024596,"remaining_time":4.491083318},
{"learn":[12522.46136],"iteration":38,"passed_time":0.1777503225,"remaining_time":4.379950255},
{"learn":[12466.96807],"iteration":39,"passed_time":0.1782796587,"remaining_time":4.27871181},
{"learn":[12408.7065],"iteration":40,"passed_time":0.1787075144,"remaining_time":4.180012349},
{"learn":[12353.71731],"iteration":41,"passed_time":0.1794975979,"remaining_time":4.094254733},
{"learn":[12296.24279],"iteration":42,"passed_time":0.1798864648,"remaining_time":4.003519694},
{"learn":[12241.75199],"iteration":43,"passed_time":0.1806138521,"remaining_time":3.924246423},
{"learn":[12189.16285],"iteration":44,"passed_time":0.1812137387,"remaining_time":3.845758232},
{"learn":[12135.5026],"iteration":45,"passed_time":0.1816851645,"remaining_time":3.767992325},
{"learn":[12082.09498],"iteration":46,"passed_time":0.182217205,"remaining_time":3.694744603},
... _solo primeras 50 líneas_
  ```
- **catboost_info\learn_error.tsv** (15.4KB)
- **catboost_info\time_left.tsv** (11.9KB)

### catboost_info\learn

- **catboost_info\learn\events.out.tfevents** (47.7KB)

### catboost_info\tmp

- **catboost_info\tmp\cat_feature_index.b3a05462-1372f298-9d6f8f1e-846fef2e.tmp** (4.0B)

## data


## models

- **models\CatBoost.joblib** (221.6KB)
- **models\LightGBM.joblib** (10.7KB)
- **models\Linear.joblib** (6.5KB)
- **models\Poly2.joblib** (11.1KB)
- **models\Stack.joblib** (460.0KB)

## utils


