# scripts/debug_check_types.py
import pandas as pd

df = pd.read_csv("data/ofertas_variables_semanticas_numerico_augmented.csv")

print("\n📊 Columnas no numéricas (tipos de datos):")
print(df.dtypes[df.dtypes == "object"])

print("\n🔍 Valores únicos en salary_numeric (muestra):")
print(df["salary_numeric"].unique()[:10])

