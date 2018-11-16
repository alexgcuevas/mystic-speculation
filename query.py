import numpy as np
import pandas as pd
import json, requests, pickle
from sqlalchemy import create_engine
import psycopg2
from scrape.scraper import connect_mystic

def get_recent_price(card_row, version=2):
    # Card attributes on which to query
    cardname = card_row['name'].replace("'","''")
    setname = card_row['set_name'].replace("'","''")
    rarity = card_row['rarity']

    # Structure query (with language)
    tablename = rarity+"_price_history_"+str(version)
    connection = connect_mystic()

    query = ("select ph.cardname, ph.setname, ph.timestamp, ph.price "
             "from {0} ph, "
             "     (select max(timestamp) as lastdate "
             "      from {0} ph2 "
             "      where ph2.cardname='{1}' and ph2.setname='{2}') mostrecent "
             "where ph.timestamp = mostrecent.lastdate ").format(tablename, cardname, setname)

    # Do the thing
    results = connection.execute(query)
    connection.close()
    for r in results:
        return (r[2], r[3])

def fill_recent_prices(cards_df):
    filled_df = cards_df.copy()
    filled_df[['recent_date','recent_price']] = cards_df.apply(get_recent_price, axis=1).apply(pd.Series)
    return filled_df

def write_recent_prices(cards_df, rarities):
    for rarity in rarities:
        filled_df = fill_recent_prices(cards_df[cards_df['rarity']==rarity])
        filled_df.to_csv(path_or_buf='data/all_vintage_cards-{}_recent.csv'.format(rarity))

if __name__ == "__main__":
    cards_df = pd.read_csv('data/all_vintage_cards.csv')
    rarities = ['rare']
    write_recent_prices(cards_df, rarities)
