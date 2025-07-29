# scrapy_employment_scraper/spiders/linkedin_spider.py

import scrapy
from scrapy_playwright.page import PageCoroutine
from scrapy_employment_scraper.items import JobItem

class LinkedInSpider(scrapy.Spider):
    name = "linkedin"
    start_urls = [
        "https://www.linkedin.com/jobs/search?keywords=data%20analyst&location=Worldwide"
    ]

    custom_settings = {
        'PLAYWRIGHT_PAGE_COROUTINES': [
            PageCoroutine("wait_for_selector", ".job-card-list__title")
        ]
    }

    def parse(self, response):
        jobs = response.css(".job-card-list__title")
        for job in jobs:
            item = JobItem()
            item['platform'] = 'LinkedIn'
            item['title'] = job.css("::text").get()
            item['url'] = response.urljoin(job.css("::attr(href)").get())
            yield scrapy.Request(
                url=item['url'],
                callback=self.parse_job_detail,
                meta={"item": item, "playwright": True}
            )

    def parse_job_detail(self, response):
        item = response.meta["item"]
        item['company'] = response.css(".topcard__org-name-link::text").get()
        item['location'] = response.css(".topcard__flavor--bullet::text").get()
        item['description'] = " ".join(response.css(".description__text *::text").getall()).strip()
        item['posted_at'] = response.css(".posted-time-ago__text::text").get()
        yield item
