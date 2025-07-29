
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main(input_csv, output_csv, report_path, heatmap_path, salary_column='salary_numeric', low_threshold=0.1, high_threshold=0.9):
    print("📥 Cargando archivo...")
    df = pd.read_csv(input_csv)

    df_num = df.select_dtypes(include=np.number)
    if salary_column not in df_num.columns:
        raise ValueError(f"❌ La columna '{salary_column}' no está en el CSV.")

    print("📊 Calculando matriz de correlaciones...")
    corr = df_num.corr()

    # Variables poco correlacionadas con salario
    low_corr_features = corr[salary_column][abs(corr[salary_column]) < low_threshold].index.tolist()
    if salary_column in low_corr_features:
        low_corr_features.remove(salary_column)

    # Variables redundantes entre sí
    redundant_pairs = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > high_threshold:
                col1 = corr.columns[i]
                col2 = corr.columns[j]
                redundant_pairs.add((col1, col2))

    redundant_to_remove = set([b for a, b in redundant_pairs if a != salary_column])
    to_drop = sorted(set(low_corr_features).union(redundant_to_remove))
    df_filtered = df.drop(columns=to_drop)

    # Guardar resultados
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df_filtered.to_csv(output_csv, index=False)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🧠 Análisis de correlaciones automáticas\n")
        f.write(f"Archivo de entrada: {input_csv}\n\n")
        f.write("Variables poco correlacionadas con salario (|corr| < 0.1):\n")
        for var in low_corr_features:
            f.write(f"  - {var}: {corr[salary_column][var]:.3f}\n")

        f.write("\nVariables redundantes (|corr| > 0.9 entre sí):\n")
        for a, b in sorted(redundant_pairs):
            f.write(f"  - {a} ~ {b}: {corr.loc[a, b]:.3f}\n")

        f.write(f"\nTotal de variables eliminadas: {len(to_drop)}\n")
        f.write("Columnas eliminadas: {}\n".format(', '.join(to_drop)))
        f.write(f"\n✅ Archivo final sin columnas irrelevantes: {output_csv}")

    print(f"✅ Archivo limpio guardado: {output_csv}")
    print(f"📝 Informe generado: {report_path}")

    # Generar heatmap
    print("🖼️ Generando heatmap de correlaciones...")
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Heatmap de Correlaciones", fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    plt.savefig(heatmap_path)
    print(f"✅ Heatmap guardado como: {heatmap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV de entrada")
    parser.add_argument("--output", default="data/ofertas_filtradas.csv", help="CSV limpio de salida")
    parser.add_argument("--report", default="reports/informe_correlaciones.txt", help="Informe de correlaciones")
    parser.add_argument("--heatmap", default="reports/heatmap_correlaciones.png", help="Imagen de salida del heatmap")
    parser.add_argument("--salary-column", default="salary_numeric", help="Nombre de la columna objetivo")
    args = parser.parse_args()

    main(args.input, args.output, args.report, args.heatmap, args.salary_column)
