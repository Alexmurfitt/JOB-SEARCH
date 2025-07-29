import streamlit as st

# ⚠️ Esta línea debe ser la primera de Streamlit
st.set_page_config(page_title="Análisis de Salario en Ciencia de Datos", layout="wide")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# CONFIGURACIÓN GENERAL
# ============================
STRUCTURED_FILE = "data/ofertas_variables_semanticas.csv"
RAW_FILE = "data/raw/sample_jobs.csv"

# ============================
# CARGA DE DATOS
# ============================
@st.cache_data
def load_data():
    df_structured = pd.read_csv(STRUCTURED_FILE)
    df_raw = pd.read_csv(RAW_FILE)
    df_structured["salary"] = df_raw["salary"]  # Asignación por índice

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

    df_structured["salary_numeric"] = df_structured["salary"].apply(parse_salary)
    return df_structured

df = load_data()

# ============================
# INTERFAZ STREAMLIT
# ============================
st.title("📊 Dashboard: Formación, Skills y Salario en Ciencia de Datos")
st.sidebar.header("🎯 Filtros")

skills = [col for col in df.columns if col.startswith("skill_")]
certs = [col for col in df.columns if col.startswith("cert_")]
educs = [col for col in df.columns if col.startswith("edu_")]

selected_skills = st.sidebar.multiselect("Skills", options=skills)
selected_certs = st.sidebar.multiselect("Certificaciones", options=certs)
selected_educs = st.sidebar.multiselect("Formaciones", options=educs)

filtered_df = df.copy()
for col in selected_skills + selected_certs + selected_educs:
    filtered_df = filtered_df[filtered_df[col] == 1]

st.markdown(f"🎓 Se encontraron **{len(filtered_df)}** ofertas que cumplen los filtros seleccionados.")

if len(filtered_df) > 0:
    st.subheader("💵 Distribución Salarial")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered_df["salary_numeric"], kde=True, ax=ax, bins=10)
    ax.set_title("Distribución de Salarios")
    ax.set_xlabel("Salario promedio (USD)")
    st.pyplot(fig)

    def plot_top_entities(columns, title, top_n=10):
        counts = filtered_df[columns].sum().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index.str.replace("_", " "), ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Frecuencia")
        st.pyplot(fig)

    st.subheader("🏆 Skills más comunes")
    plot_top_entities(skills, "Top Skills")

    st.subheader("🎓 Certificaciones más comunes")
    plot_top_entities(certs, "Top Certificaciones")

    st.subheader("📚 Formaciones más comunes")
    plot_top_entities(educs, "Top Formaciones")

st.subheader("🧾 Datos Filtrados")
st.dataframe(filtered_df[["salary", "salary_numeric"] + selected_skills + selected_certs + selected_educs])
