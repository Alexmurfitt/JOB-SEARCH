
---

````markdown
# 🔍 JOB SEARCH – Sistema Inteligente de Extracción, Análisis y Recomendación de Empleo

**JOB SEARCH** es una plataforma modular y escalable de análisis de empleo basada en datos reales. Está diseñada para extraer automáticamente ofertas laborales desde múltiples portales, analizar descripciones con técnicas de NLP, predecir salarios utilizando modelos de machine learning, y recomendar formaciones con mayor retorno de inversión. Su arquitectura se basa en scraping avanzado, procesamiento semántico, aprendizaje automático e interfaces interactivas.

---

## 🎯 Objetivos del sistema

- Extraer automáticamente ofertas desde portales globales como LinkedIn o Indeed.
- Limpiar, transformar y estructurar los datos textuales con precisión semántica.
- Detectar entidades clave (puesto, empresa, tecnología, salario...).
- Clasificar y analizar ofertas con modelos ML supervisados y reglas heurísticas.
- Visualizar información clave mediante dashboards y generar informes PDF.
- Estimar el ROI de formaciones según impacto en salario esperado.
- Recomendar cursos con mayor beneficio potencial.
- *(Opcional)* Almacenar los datos en MongoDB o PostgreSQL.

---

## ⚙️ Tecnologías utilizadas

| Área                   | Herramientas principales                                                 |
|------------------------|---------------------------------------------------------------------------|
| Lenguaje principal     | Python 3.12                                                               |
| Web Scraping           | Scrapy, Playwright, BeautifulSoup, requests, proxies dinámicos           |
| Procesamiento NLP      | spaCy, transformers, re, nltk, sklearn, pandas                           |
| Extracción de entidades| NER personalizado, BERT, spaCy                                           |
| Modelado Predictivo    | XGBoost, LightGBM, Optuna, SHAP, CatBoost                                |
| Recomendación & ROI    | Algoritmo personalizado de ranking y simulación de impacto formativo     |
| Visualización          | Streamlit, Plotly, Matplotlib, Seaborn                                  |
| Backend & API          | FastAPI, Pydantic, Uvicorn                                               |
| Automatización         | Airflow, Task Scheduler, Docker *(planificado)*                         |
| Almacenamiento         | CSV, JSON, MongoDB, PostgreSQL *(opcional)*                             |
| Entornos y versiones   | .env, venv, .gitignore, Git                                              |

---

## 🧱 Estructura del Proyecto

```bash
SEARCH_JOB/
├── config/                      # Configuración, YAMLs, scripts de entorno
├── data/                        # Datos estructurados, CSVs, procesados (no versionados)
├── docs/                        # Documentación técnica y funcional
├── img/                         # Visualizaciones y recursos gráficos
├── models/                      # Modelos entrenados (NER, clasificadores)
├── reports/                     # Informes en PDF generados automáticamente
├── scripts/                     # Código fuente organizado por función
│   ├── scraping/                # Scrapers (Scrapy, Playwright)
│   ├── preprocessing/           # Limpieza de texto, normalización
│   ├── nlp/                     # Extracción de entidades (NER)
│   ├── model_training/          # Entrenamiento y validación de modelos ML
│   ├── prediction/              # Predicción y clasificación de nuevas ofertas
│   ├── feature_engineering/     # Generación y análisis de variables
│   ├── dashboard/               # Visualización en Streamlit
│   └── informe/                 # Informes PDF y visualizaciones ejecutivas
├── scrapy_employment_scraper/  # Proyecto Scrapy funcional
├── .env                         # Variables de entorno privadas
├── requirements.txt             # Dependencias del sistema
└── README.md                    # Descripción general del sistema
````

---

## 🧠 Modelos y lógica implementada

### 🔹 NER (Reconocimiento de Entidades Nombradas)

* Extracción de: empresa, tecnología, ubicación, nivel, experiencia, idiomas, salario.
* Modelos basados en `spaCy` y `transformers` (BERT).

### 🔹 Clasificación y predicción

* Clasificadores (ej. `CatBoost`) para tipos de empleo.
* Modelos regresivos (`XGBoost`, `LightGBM`) para predicción salarial.
* Optimización con `Optuna`.
* Explicabilidad mediante `SHAP`.

### 🔹 Recomendador con ROI

* Algoritmo que estima el incremento salarial tras realizar una formación.
* Simulación de escenarios y ranking de cursos con mayor retorno.

---

## 🚀 Ejecución del sistema

```bash
# 1. Clona el repositorio y crea el entorno
git clone https://github.com/Alexmurfitt/JOB-SEARCH.git
cd JOB-SEARCH
python -m venv venv_jobs
.\venv_jobs\Scripts\activate

# 2. Instala las dependencias reales del sistema
pip install -r requirements.txt

# 3. Configura tu archivo .env si vas a usar MongoDB o PostgreSQL
```

---

## 🔧 Ejecución modular

* **Scraping** → `scripts/scraping/` o `scrapy_employment_scraper/`
* **Preprocesado** → `scripts/preprocessing/`
* **NER y NLP** → `scripts/nlp/`
* **Modelado** → `scripts/model_training/`
* **Recomendación y ROI** → `scripts/prediction/`
* **Dashboard** → `scripts/dashboard/app.py`
* **Informes PDF** → `scripts/informe/generar_informe_data_analyst.py`

---

## 📤 Exportación y Visualización

* Resultados exportados a `reports/` (PDFs, gráficos).
* Interfaz opcional en Streamlit para exploración interactiva.
* Datos intermedios y finales almacenados en `data/` y `models/`.

---

## 🛡️ Exclusiones clave (.gitignore)

```gitignore
# Entornos y secretos
.venv/
.env

# Archivos pesados o no versionables
*.joblib
*.pkl
*.pt
*.bin
*.pdf
*.tsv
*.tmp
*.log
__pycache__/
models/**/tokenizer.json
models/**/vocab.txt
catboost_info/
data/
artifacts/
```
---

# ✅ Estado del Proyecto: Plataforma de Recomendación Formativa Basada en Empleo Real

---

## 🟢 Fase 1. Análisis, planificación y diseño

| Tarea                                                                 | Estado         |
|-----------------------------------------------------------------------|----------------|
| Análisis del objetivo (identificar formaciones con mayor impacto salarial) | ✅ Completado |
| Investigación de fuentes de datos (plataformas de empleo globales)   | ✅ Completado   |
| Selección de arquitectura general del sistema                        | ✅ Completado   |
| Definición de flujos de trabajo y módulos funcionales                | ✅ Completado   |
| Herramientas seleccionadas (Scrapy, Playwright, ML, SHAP, Streamlit…) | ✅ Completado  |

---

## 🟢 Fase 2. Ingeniería de scraping y recolección de datos

| Tarea                                                               | Estado                   |
|---------------------------------------------------------------------|--------------------------|
| Identificación de las 10 principales plataformas de empleo global   | ✅ Completado            |
| Diseño de scraper modular con tolerancia a fallos                   | ✅ Completado            |
| Scraping funcional con Playwright (dinámico, multi-plataforma)      | 🟡 En desarrollo avanzado |
| Normalización de datos extraídos (HTML → JSON/CSV limpios)          | 🟡 En desarrollo avanzado |
| Gestión de bloqueos (proxies, headers, retries, captchas)           | 🟡 En desarrollo          |
| Guardado en estructura local CSV o MongoDB                          | 🔵 Listo para implementar |

---

## 🟢 Fase 3. Procesamiento de lenguaje natural (NLP)

| Tarea                                                           | Estado                   |
|-----------------------------------------------------------------|--------------------------|
| Limpieza de descripciones (remoción HTML, stopwords, etc.)      | 🔵 Listo para implementar |
| Detección de entidades clave (skills, certs, grados)            | 🔵 Listo para implementar |
| Fine-tuning de modelo BERT o uso de spaCy/NER preentrenado      | 🔵 Listo para implementar |
| Normalización semántica (sinónimos, agrupaciones)               | 🔵 Listo para implementar |
| Generación de dataset estructurado                              | 🔵 Listo para implementar |

---

## 🟡 Fase 4. Modelado predictivo de salarios

| Tarea                                                             | Estado        |
|-------------------------------------------------------------------|---------------|
| Ingeniería de features (dummies, embeddings, experiencia, etc.)   | 🔵 Planificado |
| Entrenamiento con XGBoost, LightGBM, redes neuronales             | 🔵 Planificado |
| Optimización con Optuna / CV                                      | 🔵 Planificado |
| Validación con MAE, R²                                            | 🔵 Planificado |
| Explicabilidad con SHAP                                           | 🔵 Planificado |
| Ensemble final con modelos combinados                             | 🔵 Planificado |

---

## 🟢 Fase 5. Recomendación personalizada y ROI

| Tarea                                                              | Estado                   |
|--------------------------------------------------------------------|--------------------------|
| Simulación de escenarios (añadir formaciones y estimar incremento) | 🔵 Listo para implementar |
| Cálculo de ROI y eficiencia de cada formación                      | 🔵 Listo para implementar |
| Generación automática de ranking personalizado                     | 🔵 Listo para implementar |

---

## 🟢 Fase 6. Visualización, reporting y API

| Tarea                                            | Estado                   |
|--------------------------------------------------|--------------------------|
| Diseño del dashboard en Streamlit                | 🔵 Listo para implementar |
| Visualización de resultados y simulaciones       | 🔵 Listo para implementar |
| Generación de informes PDF ejecutivos            | 🔵 Listo para implementar |
| API REST con FastAPI (predicción y sugerencia)   | 🔵 Listo para implementar |

---

## 🟢 Fase 7. Despliegue, automatización y monitoreo

| Tarea                                               | Estado        |
|-----------------------------------------------------|---------------|
| Contenerización con Docker                          | 🔵 Planificado |
| Planificación diaria con Airflow / Task Scheduler   | 🔵 Planificado |
| Sistema de logs y errores                           | 🔵 Planificado |
| Monitorización y retraining automático              | 🔵 Planificado |

---

## 📌 Resumen por estado

| Estado        | Tareas clave                                                                 |
|---------------|-------------------------------------------------------------------------------|
| ✅ Completado | Diseño general, selección tecnológica, definición de arquitectura, análisis semántico |
| 🟡 En desarrollo | Scraper multi-plataforma (Playwright), normalización de scraping               |
| 🔵 Listo para implementar | NLP (NER), modelado, recomendación, dashboard, API, reporting, despliegue     |
| 🔴 Pendiente   | Retraining automático, integración total de módulos                           |

---

## 🚀 Siguiente paso recomendado

Finalizar la fase de scraping robusto para las 10 plataformas, almacenar los datos en CSV o MongoDB, y comenzar inmediatamente con el módulo NLP (extracción de entidades) para preparar el dataset final de entrenamiento.




---

