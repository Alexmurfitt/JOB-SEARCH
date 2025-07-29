#!/usr/bin/env python
# scripts/extract_ner_errors.py

import os
import json
import csv
import pickle
import argparse
import torch
from transformers import BertForTokenClassification, BertTokenizerFast

def main():
    parser = argparse.ArgumentParser(
        description="Extrae de la validación las oraciones mal etiquetadas por el modelo NER"
    )
    parser.add_argument("--model_dir",   type=str, required=True,
                        help="Directorio con el modelo entrenado y labels.json")
    parser.add_argument("--val_texts",   type=str, required=True,
                        help="Pickle con lista de textos de validación")
    parser.add_argument("--val_tags",    type=str, required=True,
                        help="Pickle con lista de secuencias de etiquetas o de IDs de validación")
    parser.add_argument("--labels_json", type=str, required=True,
                        help="labels.json (mapping label→id) dentro de model_dir")
    parser.add_argument("--output_csv",  type=str, required=True,
                        help="CSV de salida con columnas text,tags para reanotar")
    args = parser.parse_args()

    # 1) Cargar mapping label→id y construir id→label
    if not os.path.isfile(args.labels_json):
        raise FileNotFoundError(f"No encuentro labels.json en {args.labels_json}")
    label2id = json.load(open(args.labels_json, "r", encoding="utf8"))
    id2label = {v: k for k, v in label2id.items()}

    # 2) Cargar modelo y tokenizer
    print(f"Cargando modelo desde {args.model_dir} …")
    model = BertForTokenClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model.eval()

    # 3) Cargar datos de validación
    texts     = pickle.load(open(args.val_texts, "rb"))
    raw_tags  = pickle.load(open(args.val_tags,  "rb"))

    # 4) Abrir CSV de salida
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["text", "tags"])  # cabecera

        # 5) Para cada muestra, inferir y comparar
        for sentence, gold in zip(texts, raw_tags):
            tokens = sentence.split()

            # inferencia
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(0)

            # reconstruir predicciones
            word_ids = enc.word_ids(batch_index=0)
            preds = []
            prev_wid = None
            for idx, wid in enumerate(word_ids):
                if wid is None or wid == prev_wid:
                    continue
                pred_id = logits[idx].argmax().item()
                preds.append(id2label[pred_id])
                prev_wid = wid

            # reconstruir etiquetas “gold” en forma de lista de strings
            if all(isinstance(x, int) for x in gold):
                gold_labels = [id2label[i] for i in gold]
            else:
                gold_labels = list(gold)

            # longitud mismatched: descartamos
            if len(preds) != len(tokens) or len(gold_labels) != len(tokens):
                continue

            # si difieren, volcarlas al CSV
            if preds != gold_labels:
                writer.writerow([sentence, " ".join(gold_labels)])

    print(f"✅ Casos de error volcados en {args.output_csv}")

if __name__ == "__main__":
    main()
