#!/usr/bin/env python
# scripts/evaluate_ner_model.py

import os
import json
import pickle
import argparse

import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from seqeval.metrics import classification_report


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo BERT-NER y muestra el reporte de seqeval"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directorio donde están labels.json, el modelo y el tokenizer"
    )
    parser.add_argument(
        "--val_texts", type=str, required=True,
        help="Pickle con la lista de oraciones de validación (strings)"
    )
    parser.add_argument(
        "--val_tags", type=str, required=True,
        help="Pickle con la lista de secuencias de etiquetas de validación (listas de strings)"
    )
    args = parser.parse_args()

    # 1) Carga mapping label→id y construye id→label
    labels_path = os.path.join(args.model_dir, "labels.json")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"No encuentro labels.json en {args.model_dir}")
    label2id = json.load(open(labels_path, "r", encoding="utf8"))
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # 2) Carga modelo y tokenizer
    print(f"Cargando modelo desde {args.model_dir} …")
    model = BertForTokenClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model.eval()

    # 3) Carga datos de validación
    raw_texts = pickle.load(open(args.val_texts, "rb"))
    val_tags   = pickle.load(open(args.val_tags,   "rb"))

    preds = []
    trues = []

    # 4) Loop de evaluación
    for sentence, gold_labels in zip(raw_texts, val_tags):
        tokens = sentence.split()
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits.squeeze(0)  # (seq_len, num_labels)
        word_ids = encoding.word_ids(batch_index=0)

        pred_labels = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            # saltar [CLS], [SEP] y sub-palabras posteriores
            if word_id is None or word_id == prev_word_id:
                continue
            pred_id = logits[idx].argmax().item()
            pred_labels.append(id2label[pred_id])
            prev_word_id = word_id

        if len(pred_labels) != len(tokens):
            raise ValueError(
                f"Tokens: {len(tokens)}, preds: {len(pred_labels)}, sentence: {sentence}"
            )

        preds.append(pred_labels)
        trues.append(gold_labels)

    # 5) Reporte
    print("\n==== Classification Report ====\n")
    print(classification_report(trues, preds, zero_division="0"))

    # 6) Imprimir errores caso a caso
    print("\n==== Casos mal etiquetados ====\n")
    for sent, true, pred in zip(raw_texts, trues, preds):
        if true != pred:
            print("SENT:", sent)
            print("TRUE:", true)
            print("PRED:", pred)
            print("---")


if __name__ == "__main__":
    main()

