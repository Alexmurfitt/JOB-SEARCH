import random
import logging
from pathlib import Path
from scrapy.exceptions import NotConfigured


class RandomUserAgentMiddleware:
    def __init__(self, user_agents):
        if not user_agents:
            raise NotConfigured("[Middleware] La lista de User-Agents está vacía.")
        self.user_agents = user_agents
        logging.info(f"[Middleware] ✅ {len(self.user_agents)} User-Agents cargados.")

    @classmethod
    def from_crawler(cls, crawler):
        path = Path(crawler.settings.get('USER_AGENT_LIST_PATH', '')).resolve()
        logging.info(f"[Middleware] Cargando User-Agents desde: {path}")
        if not path.exists():
            raise FileNotFoundError(f"[Middleware] ❌ No se encontró el archivo de User-Agents: {path}")
        with open(path, encoding="utf-8") as f:
            user_agents = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        return cls(user_agents)

    def process_request(self, request, spider):
        request.headers['User-Agent'] = random.choice(self.user_agents)


class RandomProxyMiddleware:
    def __init__(self, proxies):
        if not proxies:
            raise NotConfigured("[Middleware] La lista de proxies está vacía.")
        self.proxies = proxies
        logging.info(f"[Middleware] ✅ {len(self.proxies)} proxies cargados.")

    @classmethod
    def from_crawler(cls, crawler):
        path = Path(crawler.settings.get('ROTATING_PROXY_LIST_PATH', '')).resolve()
        logging.info(f"[Middleware] Cargando proxies desde: {path}")
        if not path.exists():
            raise FileNotFoundError(f"[Middleware] ❌ No se encontró el archivo de proxies: {path}")
        with open(path, encoding="utf-8") as f:
            proxies = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        return cls(proxies)

    def process_request(self, request, spider):
        request.meta['proxy'] = random.choice(self.proxies)
