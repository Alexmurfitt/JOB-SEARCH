import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import argparse
import os

def load_model(model_name="models/ner_model"):
    """
    Carga un modelo NER fine-tuned desde un directorio local o HuggingFace.
    """
    # Si pasas un directorio local existente, lo usa; si no, intenta descargar de HF.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if (model.device.type == "cuda") else -1
    )
    return ner_pipeline

import re

def extract_entities(text, ner_pipeline):
    """
    Extrae entidades del texto, devolviendo lista de "Etiqueta:Texto" 
    y filtrando tokens vacíos o solo puntuación.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    entities = ner_pipeline(text)
    clean = []
    for ent in entities:
        word = ent["word"].strip()
        # Descartar tokens muy cortos o solo símbolos
        if len(word) <= 1 or re.fullmatch(r"[\W_]+", word):
            continue
        clean.append(f"{ent['entity_group']}:{word}")
    return clean

def process_file(input_csv, output_csv, column, model_name):
    """
    Procesa un CSV de entrada, extrae entidades usando NER y guarda un CSV enriquecido.
    """
    print(f"🔍 Cargando modelo NER desde: {model_name}")
    ner_pipeline = load_model(model_name)

    df = pd.read_csv(input_csv)
    tqdm.pandas(desc="🔎 Extrayendo entidades")
    df['extracted_entities'] = df[column].progress_apply(lambda x: extract_entities(x, ner_pipeline))

    df.to_csv(output_csv, index=False)
    print(f"💾 Guardado dataset enriquecido en: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae entidades NER de descripciones de empleo usando un modelo Transformers"
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="CSV de entrada con descripciones de empleo")
    parser.add_argument("--output", type=str, required=True,
                        help="CSV de salida con entidades extraídas")
    parser.add_argument("--column", type=str, default="description",
                        help="Nombre de la columna de texto en el CSV")
    parser.add_argument("--model", type=str, default="models/ner_model",
                        help="Ruta local (o repo HF) del modelo NER fine-tuned")
    args = parser.parse_args()

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    process_file(args.input, args.output, args.column, args.model)
