scrapy_employment_scraper/
├── scrapy_employment_scraper/
│   ├── __init__.py
│   ├── items.py
│   ├── middlewares.py
│   ├── pipelines.py
│   ├── settings.py
│   └── spiders/
│       ├── __init__.py
│       └── linkedin_spider.py
├── scrapy.cfg
├── requirements.txt
├── README.md
├── launcher.py
├── airflow_dags/
│   └── dag_scraping_jobs.py

# airflow_dags/dag_scraping_jobs.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['admin@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'scraping_empleo_diario',
    default_args=DEFAULT_ARGS,
    description='Scraping diario de plataformas de empleo',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['scraping', 'empleo']
)

spiders = ['linkedin']  # Agrega más spiders aquí

for spider in spiders:
    BashOperator(
        task_id=f"scrapy_{spider}",
        bash_command=f"cd /ruta/a/proyecto && source venv/Scripts/activate && scrapy crawl {spider}",
        dag=dag
    )
