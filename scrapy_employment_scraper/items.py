# scrapy_employment_scraper/items.py

import scrapy

class JobItem(scrapy.Item):
    platform = scrapy.Field()
    title = scrapy.Field()
    company = scrapy.Field()
    location = scrapy.Field()
    description = scrapy.Field()
    url = scrapy.Field()
    posted_at = scrapy.Field()


