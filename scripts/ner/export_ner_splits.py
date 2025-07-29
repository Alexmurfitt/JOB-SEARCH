#!/usr/bin/env python
# scripts/export_ner_splits.py

import os
import json
import csv
import argparse
import random
from pathlib import Path

def load_jsonl(path):
    """
    Carga un JSONL donde cada objeto puede ocupar varias líneas:
    vamos acumulando líneas hasta que el número de '{' == número de '}',
    entonces hacemos json.loads() sobre el bloque completo.
    """
    buf = ""
    opens = 0
    closes = 0
    with open(path, "r", encoding="utf8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            buf += line
            opens  += line.count("{")
            closes += line.count("}")
            if opens > 0 and opens == closes:
                yield json.loads(buf)
                buf = ""
                opens = closes = 0
    # si quedó algo en buf, intenta un último parse
    if buf.strip():
        yield json.loads(buf)

def to_csv(input_jsonl, out_csv):
    """Lee todos los registros y los escribe en un CSV con columnas text y tags."""
    records = list(load_jsonl(input_jsonl))
    if not records:
        raise ValueError(f"No se cargó ningún registro de {input_jsonl}")
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "tags"])
        for rec in records:
            # suponemos que rec tiene rec["tokens"] y rec["labels"]
            text = " ".join(rec["tokens"])
            tags = " ".join(rec["labels"])
            writer.writerow([text, tags])
    print(f"Escrito {out_csv} ({len(records)} oraciones)")

def main():
    parser = argparse.ArgumentParser(description="Exporta JSONL NER a CSV y parte train/val")
    parser.add_argument("--input",   required=True, help="ruta a tu ner_dataset.jsonl")
    parser.add_argument("--outdir",  required=True, help="directorio donde caerán train.csv y val.csv")
    parser.add_argument("--val_pct", type=float, default=0.2, help="porcentaje de validación")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_csv = Path(args.outdir) / "ner_all.csv"
    to_csv(args.input, all_csv)

    # lee todo, baraja y parte
    with open(all_csv, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    random.shuffle(rows)
    # garantizar al menos 1 en val si hay más de 1 registro
    n_val = max(1, int(len(rows) * args.val_pct)) if len(rows) > 1 else 0
    rows_val   = rows[:n_val]
    rows_train = rows[n_val:]

    def write_split(split, rows_split):
        path = Path(args.outdir) / f"ner_{split}.csv"
        with open(path, "w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "tags"])
            for r in rows_split:
                writer.writerow([r["text"], r["tags"]])
        print(f"Escrito {path} ({len(rows_split)} oraciones)")

    write_split("train", rows_train)
    write_split("val",   rows_val)

if __name__ == "__main__":
    main()
