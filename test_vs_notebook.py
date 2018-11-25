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
from query import *


""" Testing plotting over time """
connection = connect_mystic()

# Plots Mythic history
def plot_price_history(rarity, version=2):
    history_df = get_price_history(rarity, version)
    def time_bomb(row):
        date = pd.Timestamp.utcfromtimestamp(int(row['timestamp'])/1000)
        row['year'] = date.year
        row['month'] = date.month
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

""" Loading standard seasonality into database """

seasons_df = pd.read_csv("data/set_seasonality.csv")
dates_df = pd.read_csv("data/set_dates.csv")
seasons_df.fillna(0, inplace=True)
seasons_df.head()
dates_df.head()

"""
Load average price by season
"""
connection = connect_mystic()
rarity = 'rare'
tablename = rarity+"_price_history_2"
seasons = np.array(pd.read_csv("data/season_dates.csv"))
season_prices = avg_price_by_season(seasons, tablename)
season_prices.drop(columns=['cardname','setname'], inplace=True)
plt.plot(season_prices.sum())
plt.show()
    #Upgrade avg to time-weighted average after MVP working
    #try median?
connection.close()
season_prices['s29']
seasons

"""
Load time-weighted average price by season
"""


rarity = 'rare'
tablename = rarity+"_price_history_2"
seasons = np.array(pd.read_csv("data/season_dates.csv"))

c=1000000000
connection = connect_mystic()
seasons_df = pd.DataFrame(columns=['cardname','setname'])
for season in seasons:
    # to timestamp
    start = str(int(pd.Timestamp(season[0]).value/c))
    end = str(int(pd.Timestamp(season[1]).value/c))

    # add season bookends to price history, calculate leads and diffs
    query = ("with add_season_bookends as "
             "(select ph.cardname, ph.setname, {START} as timestamp, ph.price "
             "from {TABLENAME} ph, "
             "     (select ph2.cardname, ph2.setname, max(ph2.timestamp) as lastdate "
             "      from {TABLENAME} ph2 "
             "      where ph2.timestamp < {START} "
             "      group by ph2.cardname, ph2.setname) ss "
             "where ph.timestamp = ss.lastdate "
             "and ph.cardname = ss.cardname "
             "and ph.setname = ss.setname "
             "union "
             "select ph.cardname, ph.setname, {END} as timestamp, ph.price "
             "from {TABLENAME} ph, "
             "     (select ph2.cardname, ph2.setname, max(ph2.timestamp) as lastdate "
             "      from {TABLENAME} ph2 "
             "      where ph2.timestamp < {END} "
             "      group by ph2.cardname, ph2.setname) ss "
             "where ph.timestamp = ss.lastdate "
             "and ph.cardname = ss.cardname "
             "and ph.setname = ss.setname "
             "union "
             "select * from {TABLENAME} "
             "where cast(timestamp as float)/1000 >= {START} "
             "  and cast(timestamp as float)/1000 <= {END}) "
             " "
             ",timeleads as "
             "(select *, lead(timestamp) over (order by timestamp) timelead "
             "from add_season_bookends) "
             " "
             ",diffs as "
             "(select *, datediff(day, timestamp, timelead) as daydiff "
             "from timeleads) "
             " "
             "select cardname, setname, sum(daydiff*price)/sum(day_diff) as s{SEASON}"
             " from diffs").format(START=start, END=end, TABLENAME=tablename, SEASON=season[2])
    season_df = pd.read_sql(query, connection)
    seasons_df = seasons_df.merge(season_df, on=['cardname','setname'], how='outer')
connection.close()
seasons_df.drop(columns=['cardname','setname'], inplace=True)
plt.plot(seasons_df.sum())
plt.show()


    # get timeleads

    # calculate diff

    # calculate weighted average


    query = ("select cardname, setname, avg(price) as s{3} "
                "from {0} "
                "where cast(timestamp as float)/1000 > {1} and cast(timestamp as float)/1000 < {2} "
                "group by cardname, setname ").format(tablename,start,end,season[2])
    season_df = pd.read_sql(query, connection)
    seasons_df = seasons_df.merge(season_df, on=['cardname','setname'], how='outer')
connection.close()
return seasons_df