from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import pandas as pd
import os

# ========= CONFIGURACIÓN =========
OUTPUT_PDF = "reports/informe_modelado_ml.pdf"
STRUCTURED_FILE = "data/ofertas_variables_semanticas.csv"
RAW_FILE = "data/raw/sample_jobs.csv"
IMG_PATH = "reports/figures/feature_importance.png"

def parse_salary(s):
    if pd.isna(s): return None
    s = str(s).replace("$", "").replace(",", "").strip()
    if "-" in s:
        try:
            low, high = s.split("-")
            return (float(low) + float(high)) / 2
        except:
            return None
    try:
        return float(s)
    except:
        return None

def generar_informe():
    # =====================
    # CARGA Y PREPROCESO
    # =====================
    df = pd.read_csv(STRUCTURED_FILE)
    df_raw = pd.read_csv(RAW_FILE)

    df["salary"] = df_raw["salary"]
    df["salary_numeric"] = df["salary"].apply(parse_salary)

    total = len(df)
    media = round(df["salary_numeric"].mean(), 2)
    minimo = round(df["salary_numeric"].min(), 2)
    maximo = round(df["salary_numeric"].max(), 2)
    variables = len([col for col in df.columns if col.startswith(("skill_", "cert_", "edu_"))])

    # =====================
    # CREACIÓN DEL PDF
    # =====================
    c = canvas.Canvas(OUTPUT_PDF, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2.5*cm, h - 2.5*cm, "📊 Informe Final: Modelado ML y Variables Salariales")

    c.setFont("Helvetica", 11)
    c.drawString(2.5*cm, h - 3.5*cm, "Este informe resume los resultados del modelado predictivo sobre salario.")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.5*cm, h - 5*cm, "📌 Resumen de Datos:")

    c.setFont("Helvetica", 10)
    c.drawString(3*cm, h - 6*cm, f"Total de ofertas analizadas: {total}")
    c.drawString(3*cm, h - 6.6*cm, f"Salario promedio: ${media}")
    c.drawString(3*cm, h - 7.2*cm, f"Rango salarial: ${minimo} – ${maximo}")
    c.drawString(3*cm, h - 7.8*cm, f"Número de variables predictoras: {variables}")

    if os.path.exists(IMG_PATH):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2.5*cm, h - 9.5*cm, "🎯 Importancia de Variables según XGBoost + SHAP")
        c.drawImage(IMG_PATH, 3*cm, h - 21*cm, width=12*cm, preserveAspectRatio=True)

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2.5*cm, 2*cm, "Generado automáticamente con ReportLab – Proyecto Analítica Laboral 2025")

    c.showPage()
    c.save()
    print(f"✅ Informe PDF generado correctamente: {OUTPUT_PDF}")

if __name__ == "__main__":
    generar_informe()
