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
