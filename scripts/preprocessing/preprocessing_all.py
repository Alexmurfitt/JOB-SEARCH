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
