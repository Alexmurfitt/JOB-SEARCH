#!/usr/bin/env python
# scripts/infer_ner.py

import argparse
import json

from transformers import BertForTokenClassification, BertTokenizerFast
import torch

def tag_sentence(model, tokenizer, id2label, sentence: str):
    tokens = sentence.split()
    inputs = tokenizer(tokens,
                       is_split_into_words=True,
                       return_tensors="pt",
                       padding=False,
                       truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.argmax(-1).squeeze().tolist()
    # No usamos subpalabras: pedimos un label por token original
    word_ids = inputs.word_ids(batch_index=0)
    result = []
    for idx, wid in enumerate(word_ids):
        if wid is not None and wid < len(tokens):
            result.append((tokens[wid], id2label[preds[idx]]))
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia NER con HuggingFace BERT"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Carpeta del modelo (contiene labels.json, config, pytorch_model.bin, tokenizer)")
    parser.add_argument("sentence", type=str,
                        help="Frase a etiquetar (en español)")
    args = parser.parse_args()

    # Carga
    model = BertForTokenClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model.eval()

    label2id = json.load(open(f"{args.model_dir}/labels.json", "r", encoding="utf8"))
    id2label = {i: lbl for lbl, i in label2id.items()}

    tags = tag_sentence(model, tokenizer, id2label, args.sentence)
    for token, tag in tags:
        print(f"{token:15} → {tag}")

if __name__ == "__main__":
    main()
