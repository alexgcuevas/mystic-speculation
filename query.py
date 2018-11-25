import numpy as np
import pandas as pd
import json, requests, pickle
from sqlalchemy import create_engine
import psycopg2

def get_recent_price(card_row, version=2):
    '''DEPRECATED'''
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
             "where ph.timestamp = mostrecent.lastdate "
             "and ph.cardname='{1}' and ph.setname='{2}' ").format(tablename, cardname, setname)

    # Do the thing
    results = connection.execute(query)
    connection.close()
    for r in results:
        print(r)
        print("loading {0}'s({1}) price = {2} at time {3} into dataframe".format(r[0], r[1], r[3], r[2]))
        return (r[2], r[3])

def fill_recent_prices(cards_df):
    '''DEPRECATED'''
    filled_df = cards_df.copy()
    filled_df[['recent_date','recent_price']] = cards_df.apply(get_recent_price, axis=1).apply(pd.Series)
    return filled_df

def get_recent_prices(rarity, version=2):
    tablename = rarity+'_price_history_'+str(version)
    connection = connect_mystic()

    query = ("select ph.cardname, ph.setname, ph.timestamp, ph.price "
            "from {0} ph, "
            "     (select ph2.cardname, ph2.setname, max(timestamp) as lastdate "
            "      from {0} ph2 "
            "      group by ph2.cardname, ph2.setname) mr "
            "where ph.timestamp = mr.lastdate "
            "and ph.cardname = mr.cardname "
            "and ph.setname = mr.setname ").format(tablename)

    # Do the thing
    recent_df = pd.read_sql(query, connection)
    connection.close()
    return recent_df

def get_price_history(rarity, version=2):
    tablename = rarity+'_price_history_'+str(version)
    connection = connect_mystic()

    query = ("select * from {} ").format(tablename)

    # Do the thing
    price_history_df = pd.read_sql(query, connection)
    connection.close()
    return price_history_df

def write_recent_prices(cards_df, rarities):
    # UNTESTED
    for rarity in rarities:
        print('writing {} prices to csv'.format(rarity))
        filled_df = get_recent_prices(rarity)
        merged = cards_df.merge(filled_df, on=['cardname','setname'])
        merged.to_csv(path_or_buf='data/all_vintage_cards-{}_recent.csv'.format(rarity))

def avg_price_by_season(seasons, tablename):
    c=1000000
    print(c)
    connection = connect_mystic()
    seasons_df = pd.DataFrame(columns=['cardname','setname'])
    for season in seasons:
        # to timestamp
        start = str(int(pd.Timestamp(season[0]).value/c))
        end = str(int(pd.Timestamp(season[1]).value/c))
        query = ("select cardname, setname, avg(price) as s{3} "
                 "from {0} "
                 "where cast(timestamp as float) > {1} and cast(timestamp as float) < {2} "
                 "group by cardname, setname ").format(tablename,start,end,season[2])
        season_df = pd.read_sql(query, connection)
        seasons_df = seasons_df.merge(season_df, on=['cardname','setname'], how='outer')
    connection.close()
    return seasons_df

def w_avg_price_by_season(seasons, tablename):
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

    return seasons_df

def plot_standard_trends():
    std_seasons = pd.read_csv('data/standard_seasonality.csv')
    std_seasons.set_index('setname', inplace=True)

    def seasonal_mask(row):
        for season in seasons:
            row[season] = row[season]*std_seasons.loc[row['setname']][season]
        return row

    rarities = ['mythic','rare', 'uncommon', 'common']
    standard_price_sums=pd.Series(0, index=std_seasons.columns)

    for rarity in rarities:
        seasonal_prices = pd.read_csv('data/all_vintage_cards-{}_seasonal_avg.csv'.format(rarity))
        seasonal_prices.drop(columns='Unnamed: 0',inplace=True)
        seasons = set(seasonal_prices.columns) and set(std_seasons.columns)
        standard_prices = seasonal_prices.apply(seasonal_mask, axis=1)
        sums = standard_prices.drop(columns=['cardname','setname']).sum()
        standard_price_sums = standard_price_sums+sums
        plt.plot(sums, label=rarity)

    plt.plot(standard_price_sums, label='total')
    plt.legend()
    plt.show()



def connect_mystic():
    '''
    Connects to mystic-speculation database, returns connection object
    Output:
        SqlAlchemy PostgreSQL connection object to mystic speculation database
    '''
    # Define database info
    hostname = 'mystic-speculation.cwxojtlggspu.us-east-1.rds.amazonaws.com'
    port = '5432'
    dbname = 'mystic_speculation'

    # load username and pw information for database
    with open('scrape/login.txt', 'r') as login_info:
        username = login_info.readline().strip()
        password = login_info.readline().strip()

    # connect to database with sqlalchemy engine
    db_string = 'postgres://{0}:{1}@{2}:{3}/{4}'.format(username, password, hostname, port, dbname)
    engine = create_engine(db_string)
    connection = engine.connect()
    return connection

if __name__ == "__main__":
    cards_df = pd.read_csv('data/all_vintage_cards.csv')
    rarities = ['mythic','rare', 'uncommon', 'common']
    seasons = np.array(pd.read_csv("data/season_dates.csv"))
    for rarity in rarities:
        tablename = rarity+"_price_history_2"
        print("Starting seasonal w-avg price query for {} cards".format(rarity))
        seasonal_price_history = w_avg_price_by_season(seasons, tablename)
        print("Writing seasonal {} prices to csv".format(rarity))
        seasonal_price_history.to_csv(path_or_buf='data/all_vintage_cards-{}_seasonal_avg.csv'.format(rarity))