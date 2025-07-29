#!/usr/bin/env python
# scripts/prepare_doccano_input.py

import csv
import json
import argparse
from pathlib import Path
import re

def find_spans(text, tokens, tags):
    """
    Dado un texto, su lista de tokens y sus tags,
    construye la lista de spans [(start,end,label),…] para Doccano.
    """
    spans = []
    cursor = 0

    for tok, tag in zip(tokens, tags):
        # buscar la siguiente aparición de tok a partir de cursor
        pattern = re.escape(tok)
        m = re.search(pattern, text[cursor:], flags=re.UNICODE)
        if not m:
            continue
        start = cursor + m.start()
        end   = cursor + m.end()
        cursor = end

        if tag != "O":
            # extraer la parte tras el guión (B-XXX → XXX)
            label = tag.split("-", 1)[1] if "-" in tag else tag
            spans.append([start, end, label])

    return spans

def main():
    parser = argparse.ArgumentParser(
        description="Convierte un CSV (text,tags) a JSONL para Doccano"
    )
    parser.add_argument(
        "--input_csv",  type=Path, required=True,
        help="CSV con columnas: text,tags (tags separadas por espacios)"
    )
    parser.add_argument(
        "--output_jsonl", type=Path, required=True,
        help="Dónde escribir el JSONL para Doccano"
    )
    args = parser.parse_args()

    # validaciones
    if not args.input_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {args.input_csv}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open(newline="", encoding="utf8") as f_in, \
         args.output_jsonl.open("w", encoding="utf8") as f_out:

        reader = csv.DictReader(f_in)
        required = {"text", "tags"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"El CSV debe tener las columnas {required}, "
                f"pero encontró: {reader.fieldnames}"
            )

        for row in reader:
            text = row["text"].strip()
            if not text:
                continue

            tags   = row["tags"].strip().split()
            tokens = text.split()

            if len(tokens) != len(tags):
                print(
                    f"⚠️ Tokens/tags mismatch en línea {reader.line_num}:",
                    f"{len(tokens)} tokens vs {len(tags)} tags"
                )
                continue

            spans = find_spans(text, tokens, tags)
            record = {"text": text, "labels": spans}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Generado JSONL para Doccano en: {args.output_jsonl}")

if __name__ == "__main__":
    main()
