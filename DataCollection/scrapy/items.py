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
