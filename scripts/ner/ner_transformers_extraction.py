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
