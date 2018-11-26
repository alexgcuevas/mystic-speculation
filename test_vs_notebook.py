import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.patheffects as pe
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

c=1000000
connection = connect_mystic()
seasons_df = pd.DataFrame(columns=['cardname','setname'])

for season in seasons:
    # to timestamp
    start = str(int(pd.Timestamp(season[0]).value/c))
    end = str(int(pd.Timestamp(season[1]).value/c))

    # add season bookends to price history, calculate leads and diffs
    query = ("with add_season_bookends as "
             "(select ph.cardname, ph.setname, '{START}' as timestamp, ph.price "
             "from {TABLENAME} ph, "
             "     (select ph2.cardname, ph2.setname, max(ph2.timestamp) as lastdate "
             "      from {TABLENAME} ph2 "
             "      where cast(ph2.timestamp as float) < {START} "
             "      group by ph2.cardname, ph2.setname) ss "
             "where ph.timestamp = ss.lastdate "
             "  and ph.cardname = ss.cardname "
             "  and ph.setname = ss.setname "
             "union "
             "select ph.cardname, ph.setname, '{END}' as timestamp, ph.price "
             "from {TABLENAME} ph, "
             "     (select ph2.cardname, ph2.setname, max(ph2.timestamp) as lastdate "
             "      from {TABLENAME} ph2 "
             "      where cast(ph2.timestamp as float) < {END} "
             "      group by ph2.cardname, ph2.setname) ss "
             "where ph.timestamp = ss.lastdate "
             "  and ph.cardname = ss.cardname "
             "  and ph.setname = ss.setname "
             "union "
             "select cardname, setname, timestamp, price "
             "from {TABLENAME} "
             "where cast(timestamp as float) >= {START} "
             "  and cast(timestamp as float) <= {END}) "
             " "
             ",timeleads as "
             "(select *, lead(timestamp) over (partition by cardname, setname order by timestamp) timelead "
             "from add_season_bookends) "
             " "
             ",diffs as "
             "(select *, date_part('day', to_timestamp(cast(timelead as float)) - to_timestamp(cast(timestamp as float))) as daydiff "
             "from timeleads) "
             " "
             "select cardname, setname, sum(daydiff*price)/sum(daydiff) as s{SEASON} "
             "from diffs "
             "group by cardname, setname ").format(START=start, END=end, TABLENAME=tablename, SEASON=season[2])
    season_df = pd.read_sql(query, connection)
    seasons_df = seasons_df.merge(season_df, on=['cardname','setname'], how='outer')

seasons_df.drop(columns=['cardname','setname'], inplace=True)

seasons_df.head()
plt.plot(seasons_df.sum())
plt.show()

""" testing seasonal avg price writing for mythics"""
rarities = ['mythic','rare', 'uncommon', 'common']
for rarity in rarities:
    seasonal_averages = pd.read_csv('data/all_vintage_cards-{}_seasonal_avg.csv'.format(rarity))
    plt.plot(seasonal_averages.drop(columns=['cardname','setname','Unnamed: 0']).sum(), label=rarity)

plt.legend()
plt.show()

""" 
    Outlier Testing
"""

plot_standard_trends()

rarities = ['mythic','rare','uncommon','common']
clean_seasonal_price_outliers(rarities)

# Plot all cards

rarities = ['mythic']
plot_all_cards(rarities)

""" Debug price """
# >>> See unit_tests

plot_standard_market_size()
plot_all_cards()
plot_all_standard_cards()
""" 
