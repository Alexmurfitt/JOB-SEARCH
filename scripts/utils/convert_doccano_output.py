#!/usr/bin/env python
# scripts/convert_doccano_output.py

import json
import pickle
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(
        description="Convierte la salida JSONL de Doccano a pickles para entrenamiento NER"
    )
    parser.add_argument(
        "--input_jsonl", type=str, required=True,
        help="JSONL exportado de Doccano con campos 'text' y 'labels'"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directorio donde volcar los pickles"
    )
    parser.add_argument(
        "--split_pct", type=float, default=0.8,
        help="Porcentaje para train/val split (por defecto 0.8)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    tags  = []

    # Cargamos JSONL anotado
    with open(args.input_jsonl, 'r', encoding='utf8') as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            labels = rec["labels"]  # lista de [start,end,label]
            # Reconstruir tags por token
            tokens = text.split()
            char_positions = []
            idx = 0
            for tok in tokens:
                start = text.find(tok, idx)
                end = start + len(tok)
                char_positions.append((start, end))
                idx = end
            token_tags = ["O"] * len(tokens)
            for span in labels:
                s, e, label = span
                # buscamos qué token(es) cubre este span
                for i, (ts, te) in enumerate(char_positions):
                    if ts == s and te <= e:
                        prefix = "B-"
                    elif ts >= s and te <= e:
                        prefix = "I-"
                    else:
                        continue
                    token_tags[i] = prefix + label
            texts.append(text)
            tags.append(token_tags)

    # Split train/val
    tr_texts, vl_texts, tr_tags, vl_tags = train_test_split(
        texts, tags, train_size=args.split_pct, random_state=42
    )

    # Guardar pickles
    pickle.dump(tr_texts, open(output_dir/"train_texts.pkl", "wb"))
    pickle.dump(tr_tags,  open(output_dir/"train_tags.pkl",  "wb"))
    pickle.dump(vl_texts, open(output_dir/"val_texts.pkl",   "wb"))
    pickle.dump(vl_tags,  open(output_dir/"val_tags.pkl",    "wb"))

    print("✅ Generados pickles en:", output_dir)

if __name__ == "__main__":
    main()
