import pandas as pd
import os
import re

# Ruta de entrada y salida
INPUT_PATH = "data/ofertas_variables_semanticas.csv"
OUTPUT_PATH = "data/ofertas_variables_semanticas_numerico.csv"

def convertir_rango_salarial(s):
    if pd.isna(s):
        return None
    match = re.findall(r'\$?([\d,]+)', s)
    if match:
        nums = [int(x.replace(',', '')) for x in match]
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return sum(nums) / 2
    return None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Archivo no encontrado: {INPUT_PATH}")
        return

    print(f"📥 Leyendo archivo: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    if 'salary' not in df.columns:
        print("❌ Columna 'salary' no encontrada.")
        return

    print("🔄 Convirtiendo rangos salariales a valores numéricos...")
    df['salary_numeric'] = df['salary'].apply(convertir_rango_salarial)

    print(f"💾 Guardando archivo con nueva columna en: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Proceso completado correctamente.")

if __name__ == "__main__":
    main()
