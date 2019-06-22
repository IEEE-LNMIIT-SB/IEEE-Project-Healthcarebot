# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapingsymptomsdatacdItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    Symptom = scrapy.Field()
    Symptom_desc = scrapy.Field()
    OtherRelated = scrapy.Field()
    OtherRelated_desc = scrapy.Field()
    URL = scrapy.Field()
#    pass
