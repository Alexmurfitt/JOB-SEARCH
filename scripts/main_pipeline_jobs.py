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
