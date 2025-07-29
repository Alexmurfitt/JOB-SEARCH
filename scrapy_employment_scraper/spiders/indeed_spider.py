# scrapy_employment_scraper/spiders/indeed_spider.py

import scrapy
from scrapy.loader import ItemLoader
from urllib.parse import urljoin
from ..items import JobItem

class IndeedSpider(scrapy.Spider):
    name = "indeed"
    custom_settings = {
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 30000,
    }

    def start_requests(self):
        urls = [
            "https://www.indeed.com/jobs?q=data+analyst&l=remote",
        ]
        for url in urls:
            yield scrapy.Request(
                url=url,
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        {
                            "method": "wait_for_selector",
                            "kwargs": {"selector": "a[data-hide-spinner]"}
                        }
                    ],
                },
                callback=self.parse,
                errback=self.errback
            )

    def parse(self, response):
        job_cards = response.css("a[data-hide-spinner]")
        self.logger.info(f"🔍 Se encontraron {len(job_cards)} ofertas.")

        for job in job_cards:
            loader = ItemLoader(item=JobItem(), selector=job, response=response)
            loader.add_css("title", "h2 span::text")
            loader.add_css("company", "[data-testid='companyName']::text")
            loader.add_css("location", "[data-testid='text-location']::text")
            loader.add_css("description", "[data-testid='job-snippet']::text")
            loader.add_value("platform", "Indeed")

            posted_at = job.css("span.date::text").get()
            loader.add_value("posted_at", posted_at)

            relative_url = job.css("::attr(href)").get()
            if relative_url:
                full_url = urljoin(response.url, relative_url)
                loader.add_value("url", full_url)

            yield loader.load_item()

    def errback(self, failure):
        self.logger.error(f"❌ Error Playwright: {repr(failure)}")
