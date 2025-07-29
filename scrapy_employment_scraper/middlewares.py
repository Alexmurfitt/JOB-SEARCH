# scrapy_employment_scraper/middlewares.py

import random
import logging
from pathlib import Path

class RandomUserAgentMiddleware:
    def __init__(self, user_agent_list_path):
        path = Path("config") / user_agent_list_path
        with open(path, encoding="utf-8") as f:
            self.user_agents = [line.strip() for line in f if line.strip()]
        logging.info(f"[Middleware] Cargados {len(self.user_agents)} User-Agents.")

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings.get('USER_AGENT_LIST_PATH'))

    def process_request(self, request, spider):
        request.headers['User-Agent'] = random.choice(self.user_agents)


class RandomProxyMiddleware:
    def __init__(self, proxy_list_path):
        path = Path("config") / proxy_list_path
        with open(path, encoding="utf-8") as f:
            self.proxies = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        logging.info(f"[Middleware] Cargados {len(self.proxies)} proxies.")

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings.get('ROTATING_PROXY_LIST_PATH'))

    def process_request(self, request, spider):
        request.meta['proxy'] = random.choice(self.proxies)
