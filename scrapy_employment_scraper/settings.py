# scrapy_employment_scraper/settings.py

BOT_NAME = 'scrapy_employment_scraper'

SPIDER_MODULES = ['scrapy_employment_scraper.spiders']
NEWSPIDER_MODULE = 'scrapy_employment_scraper.spiders'

# Middlewares personalizados
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': 700,
    'scrapy_employment_scraper.middlewares.RandomUserAgentMiddleware': 400,
    'scrapy_employment_scraper.middlewares.RandomProxyMiddleware': 410,
}

# Playwright para contenido dinámico
DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

PLAYWRIGHT_BROWSER_TYPE = 'chromium'
PLAYWRIGHT_LAUNCH_OPTIONS = {
    'headless': True,
    'timeout': 30000,
    'args': ['--disable-blink-features=AutomationControlled']
}

# Configuración para rotación de agentes y proxies
USER_AGENT_LIST_PATH = "config/user_agents.txt"        # 🔧 Ruta corregida
ROTATING_PROXY_LIST_PATH = "config/proxies.txt"

# Pipeline para guardar en PostgreSQL (opcional)
ITEM_PIPELINES = {
    'scrapy_employment_scraper.pipelines.PostgresPipeline': 300,
}

# Tiempos, errores y logs
RETRY_TIMES = 3
DOWNLOAD_TIMEOUT = 20
ROBOTSTXT_OBEY = False
LOG_LEVEL = 'INFO'
