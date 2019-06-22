# -*- coding: utf-8 -*-
import scrapy
from ScrapingSymptomsDatacd.items import ScrapingsymptomsdatacdItem

class SymptomsdatacrawlerSpider(scrapy.Spider):
    name = 'SymptomsDataCrawler'
    allowed_domains = ['symptomchecker.webmd.com']
    
    start_urls = ['https://symptomchecker.webmd.com/single-symptom?symptom=muscle-cramps-or-spasms-(painful)&symid=282&loc=22',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=pain-or-discomfort&symid=1&loc=22',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=nausea-or-vomiting&symid=156&loc=22',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=fatigue&symid=98&loc=66',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=warm-to-touch&symid=252&loc=35',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=diarrhea&symid=72&loc=24',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=skin-rash&symid=185&loc=68',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=bruising-or-discoloration&symid=45&loc=35',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=swelling&symid=5&loc=62',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=bloating-or-fullness&symid=23&loc=24',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=sore-throat&symid=219&loc=10',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=dizziness&symid=81&loc=2',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=vaginal-discharge&symid=50&loc=35',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=distended-stomach&symid=378&loc=22',
                  'https://symptomchecker.webmd.com/single-symptom?symptom=numbness-or-tingling&symid=164&loc=50'
                  ]

    #location of csv file  ## Default is set to json and time based file name
#    custom_settings = {
#            'FEED_FORMAT' : 'json',
#            'FEED_URI' : 'data/data__1.json'
#            }
   
    def parse(self, response):
        print('Processing...' + response.url)
#        print('URL Extracted \n')
#        print("\n".join(response.css(".related_symptoms").css(".results_table").css("td").css("a").xpath('@href').extract()))
        
        ExtractedUrlPortions = response.css(".related_symptoms").css(".results_table").css("td").css("a").xpath('@href').extract()
        for i in range(0, len(ExtractedUrlPortions)):
            print("Extracted Sub Link ::::::::: ", response.urljoin(ExtractedUrlPortions[i]))
            yield scrapy.Request(response.urljoin(ExtractedUrlPortions[i]), callback = self.parse_detail_page, dont_filter= True)
        
    def parse_detail_page(self, response):
        Symptom = response.css(".intro").css('h2::text').extract()
        Symptom_desc = response.css(".intro").css('p::text').extract()
        OtherRelated = response.css(".results_list").css('a::text').extract()
        OtherRelated_desc = response.css(".results_list").css('p::text').extract()
#        print(Symptom, Symptom_desc, OtherRelated, OtherRelated_desc)
        item = ScrapingsymptomsdatacdItem()
        item['Symptom'] = Symptom
        item['Symptom_desc'] = Symptom_desc
        item['OtherRelated'] = OtherRelated
        item['OtherRelated_desc'] = OtherRelated_desc
        item['URL'] = response.url
        yield item
