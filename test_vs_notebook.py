import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, requests, pickle, time
from bs4 import BeautifulSoup
from collections import defaultdict
# trying slimit parser
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
# connect to postgresql database
from sqlalchemy import create_engine
import psycopg2
# Import my own stuff
from scrape.scraper import *

connection = connect_mystic()

# Plots Mythic history
def plot_price_history(rarity):
    tablename = rarity+'_price_history_2'
    query = ("select * from {} ").format(tablename)
    history_df = pd.read_sql(query, connection)
    def time_bomb(row):
        date = pd.Timestamp.utcfromtimestamp(int(row['timestamp'])/1000)
        row['year'] = date.year
        row['month'] = date.month
        # row['day'] = date.day
        # row['weekday'] = date.dayofweek
        row['date'] = date
        return row
    history_df = history_df.apply(time_bomb, axis=1)
    indexed = history_df.set_index('date')
    grouper = indexed.groupby(['year', 'month'])
    monthly_avg = pd.DataFrame()
    monthly_avg['avg_price'] = grouper['price'].apply(np.mean)
    monthly_avg['med_price'] = grouper['price'].apply(np.median)
    re_df = monthly_avg.reset_index()
    re_df['date'] = re_df.apply(lambda x: pd.Timestamp(year=int(x['year']), month=int(x['month']), day=1, hour=1), axis=1)
    monthly_avg = re_df.set_index('date')
    monthly_avg.head()
    plt.title('{} average price history'.format(rarity))
    plt.plot(monthly_avg['avg_price'], label='avg_price')
    plt.plot(monthly_avg['med_price'], label='med_price')
    plt.legend()
    plt.show()

plot_price_history('mythic')
plot_price_history('rare')
plot_price_history('uncommon')
plot_price_history('common')