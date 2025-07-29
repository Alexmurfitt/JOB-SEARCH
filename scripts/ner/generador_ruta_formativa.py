# generador_ruta_formativa.py

import pandas as pd
import json
import joblib
from transformers import pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from shap import TreeExplainer
import shap

# ---------------------------
# 1. Cargar modelo NER y modelos ML
# ---------------------------
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
model = joblib.load("models/salary_predictor.pkl")
shap_explainer = TreeExplainer(model)
multi_label_skills = joblib.load("models/multilabel_skills.pkl")

# ---------------------------
# 2. Definición del perfil de usuario
# ---------------------------
perfil_usuario_texto = """
Tengo 37 años, soy nativo bilingüe con doble nacionalidad (inglesa-española). He sido profesor de inglés durante 13 años.
Estudié el primer año del grado de empresariales (fundamentos de dirección, contabilidad, micro y macroeconomía, estadística, derecho, etc.).
Estudié los dos primeros años del grado de lenguas modernas (lengua, literatura, cultura inglesa, francés, narrativa audiovisual, etc.).
Tengo certificado CELTA de enseñanza de inglés a adultos.

Certificaciones:
- Oracle Autonomous Database & Machine Learning (SQL, administración, backup, recuperación, Linux, etc.).
- Curso de análisis de datos e inteligencia artificial:
  - Google Sheets, Looker Studio, MySQL
  - Python (EDA, scraping, OOP, matplotlib, etc.)
  - Machine Learning (regresión, clasificación, clustering, reducción de dimensionalidad)
  - Deep Learning (CNN, RNN, Transfer Learning)
  - Generative AI (LLMs, transformers, LangChain, RAG)
  - APIs con FastAPI, MongoDB, Docker
"""

# ---------------------------
# 3. Extraer entidades (skills) del perfil
# ---------------------------
resultados_ner = ner(perfil_usuario_texto)
skills_extraidas = list({ent['word'].lower() for ent in resultados_ner if ent['entity_group'] == 'SKILL'})
print("Skills extraídas:", skills_extraidas)

# ---------------------------
# 4. Transformación binaria
# ---------------------------
skills_binarias = multi_label_skills.transform([skills_extraidas])
skills_df = pd.DataFrame(skills_binarias, columns=multi_label_skills.classes_)

# ---------------------------
# 5. Predicción salarial actual
# ---------------------------
pred_salarial_actual = model.predict(skills_df)[0]
print(f"Salario estimado actual: {pred_salarial_actual:.2f} €")

# ---------------------------
# 6. Simulación de salario potencial añadiendo skills objetivo
# ---------------------------
skills_objetivo = [
    "python", "sql", "aws", "machine learning", "data visualization", "docker",
    "git", "cloud computing", "deep learning", "fastapi", "looker studio", "kubernetes"
]

brechas = [skill for skill in skills_objetivo if skill not in skills_extraidas]
print("Skills ausentes con alto impacto salarial:", brechas)

mejoras_df = skills_df.copy()
for skill in brechas:
    if skill in mejoras_df.columns:
        mejoras_df.at[0, skill] = 1

salario_potencial = model.predict(mejoras_df)[0]
salto_estimado = salario_potencial - pred_salarial_actual
print(f"Salario potencial si completas la formación sugerida: {salario_potencial:.2f} €")
print(f"Aumento estimado: {salto_estimado:.2f} €")

# ---------------------------
# 7. Explicabilidad con SHAP
# ---------------------------
shap_values = shap_explainer.shap_values(skills_df)
impactos = pd.DataFrame({
    'skill': skills_df.columns,
    'presente': skills_df.values[0],
    'shap_value': shap_values[0]
}).sort_values(by='shap_value', ascending=False)

print("\nRanking de impacto salarial por skill presente:")
print(impactos[impactos['presente'] == 1].head(10))

print("\nRecomendación formativa personalizada:")
for skill in brechas:
    print(f"- Aprende '{skill}': correlación salarial positiva observada en vacantes top.")
