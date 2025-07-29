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
