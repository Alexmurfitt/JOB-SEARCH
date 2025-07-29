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