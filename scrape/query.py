import numpy as np
import pandas as pd
import json, requests, pickle
from sqlalchemy import create_engine
import psycopg2

from scrape.scraper import *

def get_recent_price(card_row):
    # Card attributes on which to query
    cardname = card_row['name'].replace("'","''")
    setname = card_row['set_name'].replace("'","''")
    rarity = card_row['rarity']

    # Structure query (with language)
    tablename = rarity + "_price_history_2"
    connection = connect_mystic()

    query = ("select ph.cardname, ph.setname, ph.timestamp, ph.price "
             "from {0} ph, "
             "     (select max(timestamp) as lastdate "
             "      from {0} ph2 "
             "      where ph2.cardname='{1}' and ph2.setname='{2}') mostrecent "
             "where ph.timestamp = mostrecent.lastdate ").format(tablename, cardname, setname)

    # Do the thing
    results = connection.execute(query)
    for r in results:
        print(r)
    return (r[2], r[3])

