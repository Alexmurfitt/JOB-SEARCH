import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main(input_csv, output_image):
    print("📥 Leyendo archivo...")
    df = pd.read_csv(input_csv)

    # Filtrar solo columnas numéricas
    numeric_df = df.select_dtypes(include='number')

    if 'salary_numeric' not in numeric_df.columns:
        raise ValueError("❌ La columna 'salary_numeric' no está presente en el archivo.")

    print("📊 Calculando matriz de correlación...")
    corr = numeric_df.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Heatmap de Correlaciones", fontsize=16)
    plt.tight_layout()

    output_dir = os.path.dirname(output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_image)
    print(f"✅ Heatmap guardado como: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--output", default="reports/heatmap_correlaciones.png", help="Ruta del archivo PNG de salida")
    args = parser.parse_args()

    main(args.input, args.output)
