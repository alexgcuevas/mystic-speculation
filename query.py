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
    for rarity in rarities:
        print('writing {} prices to csv'.format(rarity))
        filled_df = get_recent_prices(rarity)
        cards_df.join(filled_df, on=[''])
        todo_df.to_csv(path_or_buf='data/all_vintage_cards-{}_recent.csv'.format(rarity))

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
    write_recent_prices(cards_df, rarities)
