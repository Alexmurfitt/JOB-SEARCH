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
