# scrapy_employment_scraper/pipelines.py

import os
import psycopg2
import logging
from dotenv import load_dotenv
from pathlib import Path
from itemadapter import ItemAdapter

# ✅ Cargar variables de entorno desde config/.env
env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
load_dotenv(dotenv_path=env_path)


class PostgresPipeline:
    def __init__(self):
        self.connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS")
        )
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id SERIAL PRIMARY KEY,
                platform TEXT,
                title TEXT,
                company TEXT,
                location TEXT,
                description TEXT,
                url TEXT UNIQUE,
                posted_at TEXT
            );
        ''')
        self.connection.commit()
        logging.info("[PostgreSQL] Tabla 'jobs' verificada o creada.")

    def process_item(self, item, spider):
        self.cursor.execute('''
            INSERT INTO jobs (platform, title, company, location, description, url, posted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING;
        ''', (
            item.get('platform'),
            item.get('title'),
            item.get('company'),
            item.get('location'),
            item.get('description'),
            item.get('url'),
            item.get('posted_at')
        ))
        self.connection.commit()
        return item

    def close_spider(self, spider):
        self.cursor.close()
        self.connection.close()
