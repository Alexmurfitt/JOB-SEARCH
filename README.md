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
DB_HOST=localhost
DB_NAME=jobs_db
DB_USER=postgres
DB_PASS=tu_contraseña
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

## 📌 Estado actual del proyecto

🔧 En fase de consolidación: limpieza de estructura, optimización del código, documentación final y despliegue remoto.

---

## 📬 Contacto

Creado y mantenido por **Alexander Murfitt Santana**.  
GitHub: [@Alexmurfitt](https://github.com/Alexmurfitt)

---

