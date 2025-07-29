# scripts/semantic_classifier.py

import pandas as pd
import argparse
import os
import ast

def classify_list(entity_strings):
    """
    Convierte una lista de strings "Etiqueta:Texto" en un dict de columnas semánticas binarias.
    """
    semantic = {}
    # Si la columna viene como string, parsearla a lista
    if isinstance(entity_strings, str):
        try:
            entity_strings = ast.literal_eval(entity_strings)
        except Exception:
            return semantic
    if not isinstance(entity_strings, list):
        return semantic

    for ent in entity_strings:
        if not ent or ':' not in ent:
            continue
        label, word = ent.split(':', 1)
        label_low = label.strip().lower()
        word_key = word.strip().lower().replace(' ', '_')
        key = None
        if label_low == 'skill':
            key = f"skill_{word_key}"
        elif label_low == 'tool':
            key = f"tool_{word_key}"
        elif label_low == 'certification':
            key = f"certification_{word_key}"
        elif label_low == 'education':
            key = f"education_{word_key}"
        elif label_low == 'language':
            key = f"language_{word_key}"
        elif label_low == 'seniority':
            key = f"seniority_{word_key}"
        if key:
            semantic[key] = 1
    return semantic

def classify_entities_df(df, col_name):
    """
    Aplica `classify_list` a cada fila y expande los dicts en columnas.
    """
    df = df.copy()
    semantic_series = df[col_name].apply(classify_list)
    semantic_df = pd.DataFrame(list(semantic_series)).fillna(0).astype(int)
    df = pd.concat([df, semantic_df], axis=1)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clasifica semánticamente entidades extraídas de ofertas de empleo'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Ruta al CSV con columna de entidades extraídas'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Ruta de salida para el CSV con columnas semánticas'
    )
    parser.add_argument(
        '--column', type=str, default='extracted_entities',
        help='Nombre de la columna con la lista de entidades'
    )
    args = parser.parse_args()

    # Leer input
    print(f"📥 Leyendo entidades desde: {args.input}")
    df = pd.read_csv(args.input)

    if args.column not in df.columns:
        raise KeyError(
            f"La columna '{args.column}' no existe en {args.input}. "
            f"Columnas disponibles: {', '.join(df.columns)}"
        )

    # Clasificar
    print("🔍 Clasificando entidades semánticamente...")
    df = classify_entities_df(df, col_name=args.column)

    # Crear directorio de output si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Guardar resultado
    df.to_csv(args.output, index=False)
    print(f"💾 Guardado dataset semántico en: {args.output}")
