# 🔎 SEARCH_JOB – Sistema Inteligente de Extracción y Clasificación de Empleo

`SEARCH_JOB` es una plataforma de análisis inteligente diseñada para **extraer, procesar, analizar y clasificar ofertas de empleo** desde múltiples fuentes web, con el objetivo de generar conocimiento estructurado y detectar patrones relevantes del mercado laboral. Este sistema combina **scraping, procesamiento lingüístico, modelado machine learning y visualización de datos**, todo orquestado en una arquitectura modular y escalable.

---

## 📌 Objetivos principales

- Automatizar la extracción de ofertas de empleo de múltiples portales web.
- Limpiar, transformar y estandarizar la información.
- Detectar y etiquetar entidades clave (título, salario, empresa, ubicación, tecnologías, etc.).
- Clasificar las ofertas según distintos modelos (ej. NLP, catboost).
- Visualizar insights relevantes y exportar reportes.
- Guardar la información estructurada en una base de datos relacional PostgreSQL.

---

## ⚙️ Tecnologías y herramientas utilizadas

| Categoría | Herramientas |
|----------|--------------|
| Lenguaje principal | `Python 3.12` |
| Scraping | `Scrapy`, `requests`, `BeautifulSoup` |
| Procesamiento de texto | `spaCy`, `NLTK`, `re` |
| Modelado ML | `CatBoost`, `Scikit-learn`, `Joblib` |
| Visualización | `Matplotlib`, `Seaborn`, `Plotly`, `PDF` |
| Base de datos | `PostgreSQL`, `psycopg2` |
| Entorno virtual | `venv_jobs` |
| Control de versiones | `Git`, `.gitignore` |
| Organización | `scripts/`, `models/`, `data/`, `reports/`, `config/`, `docs/` |
| Infraestructura | `.env` para credenciales, carpetas excluidas con `.gitignore` |

---

## 🧱 Estructura del proyecto

```
SEARCH_JOB/
├── .github/                      # Workflows o acciones automatizadas
├── .gitignore                   # Exclusiones de Git
├── artifacts/                   # Resultados intermedios o modelos
├── config/                      # Configuración y scripts de limpieza
│   └── reorganizar_estructura_jobs.ps1
├── data/                        # Conjuntos de datos estructurados
│   ├── raw/
│   ├── processed/
├── docs/                        # Documentación técnica o referencias
├── img/                         # Gráficos usados en informes o presentaciones
├── models/                      # Modelos entrenados y recursos NLP
│   └── ner_model/, ner_ng/, catboost_info/
├── reports/                     # Informes generados (PDFs, visualizaciones)
├── scripts/                     # Código fuente modular
│   ├── scraping/
│   ├── preprocessing/
│   ├── nlp/
│   ├── feature_engineering/
│   ├── model_training/
│   └── analysis/
├── scrapy_employment_scraper/  # Proyecto Scrapy para fuentes web
├── .env                         # Variables de entorno (NO subir a GitHub)
└── README.md                    # Este archivo
```

---

## 🧠 Modelos incluidos

- **NER personalizado (spaCy)**: Reconocimiento de entidades como "Empresa", "Tecnología", "Ubicación", "Salario".
- **CatBoostClassifier**: Modelo entrenado para clasificar ofertas de empleo por sector o tipo de perfil.
- **Reglas heurísticas y limpieza avanzada**: Para corrección de ruido textual y deduplicación.

---

## 🚀 Ejecución del sistema

1. Clona el repositorio y activa el entorno virtual `venv_jobs`.
2. Crea un archivo `.env` con tus credenciales PostgreSQL:

```env
DB_HOST=
DB_NAME=
DB_USER=
DB_PASS=
```

3. Ejecuta los módulos de scraping desde `scrapy_employment_scraper/`.
4. Lanza el pipeline de limpieza, modelado y análisis desde `scripts/`.

---

## 📤 Exportación y visualización

- Los resultados se exportan a PDF (`reports/`) y se pueden consultar en dashboards personalizados.
- Los modelos y datos se almacenan localmente en `models/` y `data/`.

---

## ❌ Exclusiones importantes (.gitignore)

```bash
# Entornos virtuales
venv_jobs/
.env

# Modelos pesados
models/**/tokenizer.json
models/**/vocab.txt
catboost_info/
*.joblib

# Archivos intermedios o binarios
*.pdf
*.pt
*.tsv
*.tmp
__pycache__/
```

---

✅ Estado del Proyecto: Plataforma de Recomendación Formativa Basada en Empleo Real
🟢 Fase 1. Análisis, planificación y diseño
Tarea	Estado
Análisis del objetivo (identificar formaciones con mayor impacto salarial)	✅ Completado
Investigación de fuentes de datos (plataformas de empleo globales)	✅ Completado
Selección de arquitectura general del sistema	✅ Completado
Definición de flujos de trabajo y módulos funcionales	✅ Completado
Herramientas seleccionadas (Scrapy, Playwright, ML, SHAP, Streamlit…)	✅ Completado

🟢 Fase 2. Ingeniería de scraping y recolección de datos
Tarea	Estado
Identificación de las 10 principales plataformas de empleo global	✅ Completado
Diseño de scraper modular con tolerancia a fallos	✅ Completado
Scraping funcional con Playwright (dinámico, multi-plataforma)	🟡 En desarrollo avanzado
Normalización de datos extraídos (HTML → JSON/CSV limpios)	🟡 En desarrollo avanzado
Gestión de bloqueos (proxies, headers, retries, captchas)	🟡 En desarrollo
Guardado en estructura local CSV o MongoDB	🔵 Listo para implementar

🟢 Fase 3. Procesamiento de lenguaje natural (NLP)
Tarea	Estado
Limpieza de descripciones (remoción HTML, stopwords, etc.)	🔵 Listo para implementar
Detección de entidades clave (skills, certs, grados)	🔵 Listo para implementar
Fine-tuning de modelo BERT o uso de spaCy/NER preentrenado	🔵 Listo para implementar
Normalización semántica (sinónimos, agrupaciones)	🔵 Listo para implementar
Generación de dataset estructurado	🔵 Listo para implementar

🟡 Fase 4. Modelado predictivo de salarios
Tarea	Estado
Ingeniería de features (dummies, embeddings, experiencia, etc.)	🔵 Planificado
Entrenamiento con XGBoost, LightGBM, redes neuronales	🔵 Planificado
Optimización con Optuna / CV	🔵 Planificado
Validación con MAE, R²	🔵 Planificado
Explicabilidad con SHAP	🔵 Planificado
Ensemble final con modelos combinados	🔵 Planificado

🟢 Fase 5. Recomendación personalizada y ROI
Tarea	Estado
Simulación de escenarios (añadir formaciones y estimar incremento)	🔵 Listo para implementar
Cálculo de ROI y eficiencia de cada formación	🔵 Listo para implementar
Generación automática de ranking personalizado	🔵 Listo para implementar

🟢 Fase 6. Visualización, reporting y API
Tarea	Estado
Diseño del dashboard en Streamlit	🔵 Listo para implementar
Visualización de resultados y simulaciones	🔵 Listo para implementar
Generación de informes PDF ejecutivos	🔵 Listo para implementar
API REST con FastAPI (predicción y sugerencia)	🔵 Listo para implementar

🟢 Fase 7. Despliegue, automatización y monitoreo
Tarea	Estado
Contenerización con Docker	🔵 Planificado
Planificación diaria con Airflow / Task Scheduler	🔵 Planificado
Sistema de logs y errores	🔵 Planificado
Monitorización y retraining automático	🔵 Planificado

📌 Resumen por estado
Estado	Tareas clave (resumen)
✅ Completado	Diseño general, selección tecnológica, definición de arquitectura, análisis semántico
🟡 En desarrollo	Scraper multi-plataforma (Playwright), normalización de scraping
🔵 Listo para implementar	NLP (NER), modelado, recomendación, dashboard, API, reporting, despliegue
🔴 Pendiente	Retraining automático, integración total de módulos

🚀 Siguiente paso recomendado
Finalizar la fase de scraping robusto para las 10 plataformas, almacenar los datos en CSV o MongoDB, y comenzar inmediatamente con el módulo NLP (extracción de entidades) para preparar el dataset final de entrenamiento.




---

