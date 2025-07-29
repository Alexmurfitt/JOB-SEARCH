# scrapy_employment_scraper/spiders/linkedin_spider.py

import scrapy
from scrapy.loader import ItemLoader
from ..items import JobItem

class LinkedInSpider(scrapy.Spider):
    name = "linkedin"
    custom_settings = {
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler"
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 60000,
    }

    def start_requests(self):
        url = "https://www.linkedin.com/jobs/search?keywords=data%20analyst&location=Worldwide"
        yield scrapy.Request(
            url,
            meta={
                "playwright": True,
                "playwright_page_methods": [
                    {
                        "method": "wait_for_selector",
                        "kwargs": {"selector": ".jobs-search-results__list-item"}
                    }
                ]
            },
            callback=self.parse,
            errback=self.errback
        )

    def parse(self, response):
        jobs = response.css(".jobs-search-results__list-item")
        for job in jobs:
            loader = ItemLoader(item=JobItem(), selector=job)
            loader.add_css("title", "h3.base-search-card__title::text")
            loader.add_css("company", "h4.base-search-card__subtitle::text")
            loader.add_css("location", ".job-search-card__location::text")
            loader.add_css("summary", "p.job-search-card__snippet::text")
            loader.add_css("url", "a.base-card__full-link::attr(href)")
            loader.add_value("platform", "LinkedIn")
            yield loader.load_item()

    def errback(self, failure):
        self.logger.error(f"[Playwright Error] {repr(failure)}")
