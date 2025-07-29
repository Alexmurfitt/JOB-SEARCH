#!/usr/bin/env python
# scripts/export_ner_data.py

import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def load_jsonl(path):
    """Carga un JSONL con objetos que tienen 'text' y 'tags'."""
    texts, tags = [], []
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error de JSON en línea {lineno}: {e}")
            text = obj.get("text")
            tag_seq = obj.get("tags")
            if text is None or tag_seq is None:
                raise KeyError(
                    f"Cada línea debe tener 'text' y 'tags'. "
                    f"Encontrado en línea {lineno}: claves disponibles {list(obj.keys())}"
                )
            # Aseguramos que tags es lista de enteros
            if not isinstance(tag_seq, list) or not all(isinstance(t, int) for t in tag_seq):
                raise ValueError(f"'tags' debe ser lista de enteros en línea {lineno}")
            texts.append(text)
            # Convertimos la lista de ints en un string "0 1 0 2 …"
            tags.append(" ".join(map(str, tag_seq)))
    return texts, tags

def main(args):
    texts, tags = load_jsonl(args.input)
    df = pd.DataFrame({"text": texts, "tags": tags})

    # División train/val
    train_df, val_df = train_test_split(
        df, test_size=args.val_pct, random_state=42, shuffle=True
    )

    os.makedirs(args.outdir, exist_ok=True)
    train_path = os.path.join(args.outdir, "train.csv")
    val_path   = os.path.join(args.outdir, "val.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path,   index=False, encoding='utf-8')

    print(f"✅ Generados:\n  {train_path}\n  {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exporta JSONL anotado a CSV train/val para NER."
    )
    parser.add_argument(
        "--input",    type=str, required=True,
        help="Ruta al JSONL con claves 'text' y 'tags'"
    )
    parser.add_argument(
        "--outdir",   type=str, required=True,
        help="Carpeta donde generar train.csv y val.csv"
    )
    parser.add_argument(
        "--val_pct",  type=float, default=0.2,
        help="Porcentaje (0–1) del dataset que irá a validación"
    )
    args = parser.parse_args()
    main(args)
