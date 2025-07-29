import pandas as pd
import numpy as np
import argparse
import os

def parse_salary(salary_str):
    """Convierte una cadena como '$90,000 - $110,000' a la media numérica."""
    if pd.isna(salary_str):
        return np.nan
    try:
        salary_str = salary_str.replace("$", "").replace(",", "")
        parts = salary_str.split("-")
        numbers = [float(p.strip()) for p in parts]
        return sum(numbers) / len(numbers) if numbers else np.nan
    except Exception:
        return np.nan

def generate_synthetic_row(reference_row):
    """Genera una nueva fila basada en una de ejemplo (con pequeñas variaciones)."""
    new_row = reference_row.copy()
    for col in reference_row.index:
        if isinstance(reference_row[col], (int, float)):
            noise = np.random.normal(0, 1)
            new_row[col] = reference_row[col] + noise
    return new_row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-rows", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"📊 Datos reales: {len(df)} filas")

    # Aseguramos que salary_numeric exista
    if "salary_numeric" not in df.columns:
        df["salary_numeric"] = df["salary"].apply(parse_salary)

    rows_needed = args.min_rows - len(df)
    if rows_needed > 0:
        print(f"⚠️ Dataset incompleto. Generando {rows_needed} datos sintéticos...")
        synthetic_rows = [generate_synthetic_row(df.sample(1).iloc[0]) for _ in range(rows_needed)]
        df_synthetic = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, df_synthetic], ignore_index=True)

    df.to_csv(args.output, index=False)
    print(f"✅ Dataset aumentado guardado en: {args.output} (total: {len(df)} filas)")

if __name__ == "__main__":
    main()
