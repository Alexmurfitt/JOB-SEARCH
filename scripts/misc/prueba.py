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
