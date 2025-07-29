# scraper_advanced_stack.py

"""
Sistema de scraping más avanzado, completo y sofisticado para plataformas de empleo globales
Tecnologías: Scrapy + Playwright + Proxy Rotación + PostgreSQL + Airflow
"""

import scrapy
from scrapy_playwright.page import PageCoroutine
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import random
import logging
from typing import Optional

# Middleware para rotación de proxies y user-agents
class AdvancedProxyMiddleware:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/112.0"
    ]

    PROXIES = [
        "http://username:password@proxy.smartproxy.com:10000",
        "http://username:password@proxy.smartproxy.com:10001"
    ]

    def process_request(self, request, spider):
        request.headers['User-Agent'] = random.choice(self.USER_AGENTS)
        request.meta['proxy'] = random.choice(self.PROXIES)

# Spider base con Playwright
class JobSpider(scrapy.Spider):
    name = 'job_spider'
    custom_settings = {
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'DOWNLOAD_HANDLERS': {
            'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
        },
        'DOWNLOADER_MIDDLEWARES': {
            '__main__.AdvancedProxyMiddleware': 350,
            'scrapy_playwright.middleware.PlaywrightMiddleware': 543,
        },
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {'headless': True},
        'FEED_URI': 'output/jobs.json',
        'FEED_FORMAT': 'json',
    }

    def start_requests(self):
        urls = [
            'https://www.linkedin.com/jobs/search?keywords=data%20analyst',
            'https://www.indeed.com/jobs?q=data+analyst',
            'https://www.glassdoor.com/Job/data-analyst-jobs-SRCH_KO0,13.htm',
        ]
        for url in urls:
            yield scrapy.Request(
                url,
                meta={"playwright": True, "playwright_page_coroutines": [PageCoroutine("wait_for_selector", "body")]},
            )

    def parse(self, response):
        for job in response.css('div.job_seen_beacon, div.jobCard, div.job-card-list__title'):
            yield {
                'title': job.css('h2::text, a::text').get(),
                'url': job.css('a::attr(href)').get(),
                'source': response.url,
            }

# Airflow DAG para scraping diario
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

def create_airflow_dag():
    with DAG(
        dag_id='job_scraping_pipeline',
        start_date=datetime(2024, 1, 1),
        schedule_interval='@daily',
        catchup=False,
    ) as dag:
        run_scraper = BashOperator(
            task_id='scrape_jobs',
            bash_command='scrapy runspider scraper_advanced_stack.py',
        )

    return dag

globals()['job_scraping_pipeline'] = create_airflow_dag()

# Iniciador local para debugging (opcional)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process = CrawlerProcess(get_project_settings())
    process.crawl(JobSpider)
    process.start()
