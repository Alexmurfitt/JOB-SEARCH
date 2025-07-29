# generar_informe_data_analyst.py
# Script definitivo para generar informe completo, detallado y profesional sobre formación y salario en Data Analyst

import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from collections import Counter
import spacy
import os

# ----------- CONFIGURACIÓN -----------
CSV_FILE = "data/raw/sample_jobs.csv"
OUTPUT_PDF = "informe_formacion_data_analyst_completo.pdf"
IMG_DIR = "img"
os.makedirs(IMG_DIR, exist_ok=True)

# Keywords detallados
BACHELOR_TERMS = ["bachelor of science", "bachelor’s degree", "b.sc", "degree in statistics"]
MASTER_TERMS = ["master of science", "master’s in data science", "msc data"]
PHD_TERMS = ["phd", "doctorate"]
CERTIFICATIONS = ["google data analytics", "ibm data analyst", "excel", "tableau", "power bi", "azure"]

nlp = spacy.load("en_core_web_sm")

# Función para limpiar texto incompatible con FPDF
def safe_text(text):
    return (text.replace("’", "'")
                .replace("“", '"')
                .replace("”", '"')
                .replace("–", "-")
                .replace("…", "...")
                .replace("→", "->")
                .replace("–", "-"))

# ----------- FUNCIONES -----------
def cargar_datos():
    df = pd.read_csv(CSV_FILE)
    df.dropna(subset=["description"], inplace=True)
    return df

def extraer_info(descripciones):
    b_count, m_count, p_count, c_count = Counter(), Counter(), Counter(), Counter()
    for desc in descripciones:
        doc = nlp(desc.lower())
        for sent in doc.sents:
            s = sent.text
            for b in BACHELOR_TERMS:
                if b in s:
                    b_count[b] += 1
            for m in MASTER_TERMS:
                if m in s:
                    m_count[m] += 1
            for p in PHD_TERMS:
                if p in s:
                    p_count[p] += 1
            for c in CERTIFICATIONS:
                if c in s:
                    c_count[c] += 1
    return b_count, m_count, p_count, c_count

def parse_salary(s):
    import re
    if pd.isna(s) or "$" not in s:
        return None
    nums = re.findall(r"\$([0-9,]+)", s)
    try:
        nums = [int(n.replace(",", "")) for n in nums]
        return sum(nums)/len(nums) if nums else None
    except:
        return None

def asociar_salario(df, categorias):
    df["parsed_salary"] = df["salary"].apply(parse_salary)
    resultados = []
    for _, row in df.iterrows():
        desc = row["description"].lower()
        for grupo, claves in categorias.items():
            for clave in claves:
                if clave in desc:
                    resultados.append((grupo, row["parsed_salary"]))
    df_sal = pd.DataFrame(resultados, columns=["Formacion", "Salary"])
    return df_sal.dropna()

def graficar(datos, titulo, nombre):
    df = pd.DataFrame(datos.items(), columns=["Etiqueta", "Frecuencia"])
    df = df.sort_values(by="Frecuencia", ascending=False)
    path = f"{IMG_DIR}/{nombre}.png"
    plt.figure(figsize=(8,4))
    plt.bar(df["Etiqueta"], df["Frecuencia"])
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# ----------- INFORME PDF -----------
def crear_pdf(b, m, p, c, img_b, img_m, img_p, img_c, img_sal, sal_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, safe_text("Informe Profesional: Formacion y Salario (Data Analyst)"), ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, safe_text(
        "Este informe presenta un analisis preciso y completo de las formaciones y certificaciones mas mencionadas en \
        ofertas de Data Analyst, asi como su relacion con el salario promedio."
    ))

    def seccion(titulo, datos, img):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, safe_text(titulo), ln=True)
        pdf.set_font("Arial", '', 11)
        for k, v in datos.items():
            pdf.cell(0, 8, safe_text(f" - {k}: {v}"), ln=True)
        pdf.image(img, w=180)

    seccion("1. Grados universitarios", b, img_b)
    seccion("2. Masters", m, img_m)
    seccion("3. Doctorados", p, img_p)
    seccion("4. Certificaciones", c, img_c)

    # Salario
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, safe_text("5. Salario promedio por formacion"), ln=True)
    resumen = sal_df.groupby("Formacion")["Salary"].mean().sort_values(ascending=False)
    pdf.set_font("Arial", '', 11)
    for k, v in resumen.items():
        pdf.cell(0, 8, safe_text(f" - {k}: ${v:,.2f}"), ln=True)
    pdf.image(img_sal, w=180)

    # Guardar
    pdf.output(OUTPUT_PDF)

# ----------- EJECUCIÓN -----------
if __name__ == "__main__":
    df = cargar_datos()
    b, m, p, c = extraer_info(df["description"].tolist())
    img_b = graficar(b, "Grados", "bachelors")
    img_m = graficar(m, "Masters", "masters")
    img_p = graficar(p, "Doctorados", "phds")
    img_c = graficar(c, "Certificaciones", "certs")
    sal = asociar_salario(df, {
        "Bachelor": list(b.keys()),
        "Master": list(m.keys()),
        "PhD": list(p.keys()),
        "Certificacion": list(c.keys())
    })
    img_sal = graficar(sal.groupby("Formacion")["Salary"].mean().to_dict(), "Salario promedio", "salarios")
    crear_pdf(b, m, p, c, img_b, img_m, img_p, img_c, img_sal, sal)
    print("✅ Informe generado con éxito:", OUTPUT_PDF)
