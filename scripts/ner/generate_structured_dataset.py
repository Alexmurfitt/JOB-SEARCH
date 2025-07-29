# scripts/generate_structured_dataset.py

import pandas as pd
import numpy as np
import argparse
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_salary(salary_str):
    """
    Convierte rangos salariales en string a un número promedio anual.
    """
    if not isinstance(salary_str, str):
        return np.nan
    # Reemplaza 'k' por '000' y extrae números
    cleaned = salary_str.lower().replace('k', '000')
    nums = re.findall(r"\d+\.?\d*", cleaned)
    if not nums:
        return np.nan
    vals = list(map(float, nums))
    return np.mean(vals)

def generate_features(df, salary_col, normalize_flag, tfidf_flag, tfidf_n):
    # 1) Salario
    if normalize_flag:
        df['salary_normalized'] = df[salary_col].apply(normalize_salary)
    else:
        df['salary_normalized'] = df[salary_col]

    # 2) Columnas semánticas binarias
    prefixes = ['skill_', 'tool_', 'certification_', 'education_', 'language_', 'seniority_']
    entity_cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    entity_df = df[entity_cols].fillna(0).astype(int)

    # 3) TF-IDF de descripciones
    if tfidf_flag:
        vect = TfidfVectorizer(max_features=tfidf_n, stop_words='english')
        tfidf_mat = vect.fit_transform(df['description'].astype(str))
        tfidf_df = pd.DataFrame(
            tfidf_mat.toarray(),
            columns=[f"tfidf_{w}" for w in vect.get_feature_names_out()],
            index=df.index
        )
    else:
        tfidf_df = pd.DataFrame(index=df.index)

    # 4) Otras variables
    if 'description' in df.columns:
        df['description_length'] = df['description'].astype(str).str.len()
    else:
        df['description_length'] = 0

        df['num_entities'] = entity_df.sum(axis=1)

    # 5) Concatenar todas las features
    features = pd.concat([
        df[['salary_normalized', 'description_length', 'num_entities']],
        entity_df,
        tfidf_df
    ], axis=1)

    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera dataset de features con TF-IDF opcional'
    )
    parser.add_argument(
        '--input', required=True,
        help='CSV semántico de entrada'
    )
    parser.add_argument(
        '--output', required=True,
        help='CSV de salida de features'
    )
    parser.add_argument(
        '--salary-column', dest='sal_col', default='salary',
        help='Columna salarial original'
    )
    parser.add_argument(
        '--normalize-salary', dest='normalize_flag',
        action='store_true',
        help='Normaliza salario'
    )
    parser.add_argument(
        '--tfidf', dest='tfidf_flag',
        action='store_true',
        help='Incluir TF-IDF'
    )
    parser.add_argument(
        '--tfidf-features', dest='tfidf_n', type=int,
        default=50,
        help='Número de términos TF-IDF'
    )
    args = parser.parse_args()

    # Cargar semántico
    print(f"📥 Cargando datos desde: {args.input}")
    df = pd.read_csv(args.input)

    # Generar features
    print("⚙️ Generando features...")
    features_df = generate_features(
        df,
        salary_col=args.sal_col,
        normalize_flag=args.normalize_flag,
        tfidf_flag=args.tfidf_flag,
        tfidf_n=args.tfidf_n
    )

    # Guardar
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"✅ Features guardadas en: {args.output}")
