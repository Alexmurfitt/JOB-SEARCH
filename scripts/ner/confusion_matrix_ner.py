#!/usr/bin/env python
# scripts/confusion_matrix_ner.py

import os
import json
import pickle
import argparse

import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Genera y muestra la matriz de confusión aplanada para un modelo BERT-NER"
    )
    parser.add_argument("--model_dir",  type=str, required=True, help="Directorio del modelo (incluye labels.json)")
    parser.add_argument("--val_texts",  type=str, required=True, help="Pickle con oraciones de validación")
    parser.add_argument("--val_tags",   type=str, required=True, help="Pickle con secuencias de etiquetas de validación")
    parser.add_argument("--output_png", type=str, default="confusion_matrix.png", help="Ruta donde guardar la imagen")
    args = parser.parse_args()

    # Carga labels.json
    labels_path = os.path.join(args.model_dir, "labels.json")
    label2id = json.load(open(labels_path, "r", encoding="utf8"))
    id2label = {v:k for k,v in label2id.items()}
    label_list = [label for label, _ in sorted(label2id.items(), key=lambda x: x[1])]

    # Carga modelo y tokenizer
    model = BertForTokenClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model.eval()

    # Carga datos
    texts = pickle.load(open(args.val_texts, "rb"))
    true_sequences = pickle.load(open(args.val_tags, "rb"))

    y_true_flat = []
    y_pred_flat = []

    # Inferencia
    for sent, true_labels in zip(texts, true_sequences):
        tokens = sent.split()
        enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.squeeze(0)  # (seq_len, n_labels)
        word_ids = enc.word_ids(batch_index=0)

        prev_wid = None
        preds = []
        for idx, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                continue
            pred_id = logits[idx].argmax().item()
            preds.append(id2label[pred_id])
            y_true_flat.append(true_labels[wid])
            y_pred_flat.append(id2label[pred_id])
            prev_wid = wid

    # Computar matriz y mostrar
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=label_list)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_list)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Matriz de Confusión NER (aplanada)")
    plt.tight_layout()
    plt.savefig(args.output_png)
    print(f"✅ Guardada matriz en {args.output_png}")

if __name__ == "__main__":
    main()
