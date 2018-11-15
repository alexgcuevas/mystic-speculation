from model.master_transmuter import *
from scrape.scraper import *

def scraping_progress(rarities):
    connection = connect_mystic()
    
    for rarity in rarities:
        results = connection.execute("select count(*) from {}_price_history_2".format(rarity))
        print('Number of {} price datapoints:'.format(rarity))
        for r in results:
            print(r[0])

        results = connection.execute("select count(distinct cardname) from {}_price_history_2".format(rarity))
        print('Number of {} cards recorded:'.format(rarity))
        for r in results:
            print(r[0])