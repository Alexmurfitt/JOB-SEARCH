# Proyecto: Scraper Profesional de Plataformas de Empleo
# Estructura Base con Scrapy + Playwright + PostgreSQL + Docker + Airflow (Fase 1)

# Paso 1: Estructura de carpetas y archivos principales
# Creamos la estructura base del proyecto
project_structure = {
    "scraper_jobs/": [
        "__init__.py",
        "items.py",
        "middlewares.py",
        "pipelines.py",
        "settings.py",
        "utils.py",
        "models.py",
        "spiders/": [
            "__init__.py",
            "indeed_spider.py",
            "linkedin_spider.py",
            "glassdoor_spider.py",
            "monster_spider.py",
            "infojobs_spider.py",
        ],
    ],
    "docker-compose.yml": None,
    "Dockerfile": None,
    ".env": None,
    "requirements.txt": None,
    "README.md": None,
    "dags/": ["dag_scraping_jobs.py"],
}

# Paso 2: Inicio de archivo settings.py (Scrapy con Playwright y proxies)
settings_template = """
BOT_NAME = 'scraper_jobs'

SPIDER_MODULES = ['scraper_jobs.spiders']
NEWSPIDER_MODULE = 'scraper_jobs.spiders'

ROBOTSTXT_OBEY = False
CONCURRENT_REQUESTS = 16
DOWNLOAD_DELAY = 3

DOWNLOADER_MIDDLEWARES = {
    'scrapy_playwright.middleware.PlaywrightMiddleware': 543,
    'scraper_jobs.middlewares.RotateUserAgentMiddleware': 400,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
}

PLAYWRIGHT_BROWSER_TYPE = 'chromium'
PLAYWRIGHT_LAUNCH_OPTIONS = {
    'headless': True,
    'timeout': 10000,
}

ITEM_PIPELINES = {
    'scraper_jobs.pipelines.PostgreSQLPipeline': 300,
}

PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 10000
FEED_EXPORT_ENCODING = 'utf-8'
"""

# Paso 3: requirements.txt básico (se ampliará según avance el sistema)
requirements_template = """
scrapy
scrapy-playwright
playwright
psycopg2-binary
sqlalchemy
python-dotenv
pandas
"""

# Paso 4: Dockerfile base
Dockerfile_template = """
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && playwright install

COPY . .

CMD ["scrapy", "crawl", "indeed_spider"]
"""

# Paso 5: docker-compose.yml básico con PostgreSQL
compose_template = """
version: '3.9'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: scraper_jobs
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  scraper:
    build: .
    volumes:
      - .:/app
    depends_on:
      - db
    env_file:
      - .env

volumes:
  pgdata:
"""

# Paso 6: .env
env_template = """
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=scraper_jobs
POSTGRES_HOST=db
POSTGRES_PORT=5432
"""

# Paso 7: README.md inicial
readme_template = """
# Scraper Profesional de Plataformas de Empleo

Sistema modular y robusto para extraer datos de empleo de las principales plataformas globales, combinando Scrapy, Playwright, proxies rotativos, y PostgreSQL.

## Componentes
- Scrapy (crawling)
- Playwright (renderizado JS)
- PostgreSQL (almacenamiento)
- Docker y Docker Compose
- Airflow (orquestación futura)

## Ejecución
```bash
docker-compose up --build
```

## Estructura de spiders (por plataforma)
- `spiders/indeed_spider.py`
- `spiders/linkedin_spider.py`
...

## Variables de entorno
Configura `.env` con las credenciales de tu base de datos PostgreSQL.
"""

# Resultado: base del sistema generada
import os
os.makedirs("scraper_jobs/spiders", exist_ok=True)
os.makedirs("dags", exist_ok=True)

with open("scraper_jobs/__init__.py", "w"): pass
with open("scraper_jobs/spiders/__init__.py", "w"): pass
with open("scraper_jobs/settings.py", "w") as f: f.write(settings_template)
with open("requirements.txt", "w") as f: f.write(requirements_template)
with open("Dockerfile", "w") as f: f.write(Dockerfile_template)
with open("docker-compose.yml", "w") as f: f.write(compose_template)
with open(".env", "w") as f: f.write(env_template)
with open("README.md", "w") as f: f.write(readme_template)

print("✅ Proyecto base Scrapy + Playwright + PostgreSQL generado correctamente.")
