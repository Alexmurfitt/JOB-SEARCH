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
