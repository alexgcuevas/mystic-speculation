import numpy as np
import pandas as pd
import json, requests, pickle
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.patheffects as pe
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

def get_card_price_history(card_row, version=2):
        # Card attributes on which to query
    cardname = card_row['name'].replace("'","''")
    setname = card_row['set_name'].replace("'","''")
    rarity = card_row['rarity'] 
    # Structure query (with language)
    tablename = rarity+"_price_history_"+str(version)
    connection = connect_mystic()

    query = ("select ph.cardname, ph.setname, ph.timestamp, ph.price "
             "from {0} ph "
             "where ph.cardname='{1}' and ph.setname='{2}' ").format(tablename, cardname, setname)

    # Do the thing
    results = connection.execute(query)
    connection.close()
    c=1000000
    price_history = []
    for r in results:
        time = str(pd.Timestamp.utcfromtimestamp(int(r[2])/1000))
        price_history.append([time,r[3]])
    print("loading {0}'s({1}) prices".format(cardname, setname))
    return price_history

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

def get_twavg_card(cardname, setname, seasons, tablename):
    c=1000000
    connection = connect_mystic()
    seasons_df = pd.DataFrame(columns=['cardname','setname'])
    for season in seasons:
        # to timestamp
        start = str(int(pd.Timestamp(season[0]).value/c))
        end = str(int(pd.Timestamp(season[1]).value/c))

        # add season bookends to price history, calculate leads and diffs
        query = ("with test_card as "
                "(select * from {TABLENAME} "
                "where cardname=\'{CARDNAME}\' "
                "  and setname=\'{SETNAME}\') "
                " "
                ",add_season_bookends as "
                "(select ph.cardname, ph.setname, '{START}' as timestamp, ph.price "
                "from test_card ph, "
                "     (select ph2.cardname, ph2.setname, max(ph2.timestamp) as lastdate "
                "      from test_card ph2 "
                "      where cast(ph2.timestamp as float) < {START} "
                "      group by ph2.cardname, ph2.setname) ss "
                "where ph.timestamp = ss.lastdate "
                "union "
                "select ph.cardname, ph.setname, '{END}' as timestamp, ph.price "
                "from test_card ph, "
                "     (select ph2.cardname, ph2.setname, max(ph2.timestamp) as lastdate "
                "      from test_card ph2 "
                "      where cast(ph2.timestamp as float) < {END} "
                "      group by ph2.cardname, ph2.setname) ss "
                "where ph.timestamp = ss.lastdate "
                "union "
                "select cardname, setname, timestamp, price "
                "from test_card "
                "where cast(timestamp as float) >= {START} "
                "  and cast(timestamp as float) <= {END}) "             
                " "
                ",leads as "
                "(select *, lead(timestamp) over (partition by cardname, setname order by timestamp) timelead, "
                "           lead(price) over (partition by cardname, setname order by timestamp) pricelead "
                "from add_season_bookends) "
                " "
                ",diffs as "
                "(select *, date_part('day', to_timestamp(cast(timelead as float)) - to_timestamp(cast(timestamp as float))) as daydiff "
                "from leads) "
                " "
                "select cardname, setname, sum(daydiff*(price+pricelead)/2)/sum(daydiff) as s{SEASON} "
                "from diffs "
                "group by cardname, setname ").format(START=start,
                                                      END=end,
                                                      TABLENAME=tablename,
                                                      CARDNAME=cardname,
                                                      SETNAME=setname,
                                                      SEASON=season[2])
        season_df = pd.read_sql(query, connection)
        seasons_df = seasons_df.merge(season_df, on=['cardname','setname'], how='outer')

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
                ",leads as "
                "(select *, lead(timestamp) over (partition by cardname, setname order by timestamp) timelead, "
                "           lead(price) over (partition by cardname, setname order by timestamp) pricelead "
                "from add_season_bookends) "
                " "
                ",diffs as "
                "(select *, date_part('day', to_timestamp(cast(timelead as float)) - to_timestamp(cast(timestamp as float))) as daydiff "
                "from leads) "
                " "
                "select cardname, setname, sum(daydiff*(price+pricelead)/2)/sum(daydiff) as s{SEASON} "
                "from diffs "
                "group by cardname, setname ").format(START=start, END=end, TABLENAME=tablename, SEASON=season[2])
        season_df = pd.read_sql(query, connection)
        seasons_df = seasons_df.merge(season_df, on=['cardname','setname'], how='outer')

    return seasons_df

def get_standard_prices(rarity, std_sets):
    seasonal_prices = pd.read_csv('data/clean_cards-{}_seasonal_avg.csv'.format(rarity))
    seasons = set(seasonal_prices.columns) and set(std_sets.columns)

    def standard_mask(row):
        for season in seasons:
            row[season] = row[season]*std_sets.loc[row['setname']][season]
        return row

    return seasonal_prices.apply(standard_mask, axis=1)

def month_formatter(ax):
    months = MonthLocator(range(1, 13), bymonthday=1, interval=3)
    monthsFmt = DateFormatter("%b '%y")
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(months)

def get_standard_format():
    std_sets = pd.read_csv('data/standard_seasonality.csv')
    std_sets.set_index('setname', inplace=True)
    dates_df = pd.read_csv('data/season_dates.csv')
    std_dates = pd.to_datetime(dates_df['end_date'].values)
    return std_sets, std_dates

def get_standard_format_real():
    std_sets = pd.read_csv('data/standard_seasonality_real.csv')
    std_sets.set_index('setname', inplace=True)
    dates_df = pd.read_csv('data/season_dates_real.csv')
    std_dates = pd.to_datetime(dates_df['end_date'].values)
    return std_sets, std_dates

def plot_standard_market_size(rarities = ['mythic','rare', 'uncommon', 'common'],
                              color_dict = {'mythic':'r', 'rare':'goldenrod', 'uncommon':'grey', 'common':'k'}):
    # Define the standard format
    std_sets, std_dates = get_standard_format_real()
    
    print(std_dates)

    # Plot setup
    standard_total=pd.Series(0, index=std_sets.columns)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Calculate standard sums & plot
    for rarity in rarities:
        print("Getting standard prices for {}s".format(rarity))
        standard_prices = get_standard_prices(rarity, std_sets)
        rarity_sum = standard_prices.drop(columns=['cardname','setname']).sum()
        print("Plotting standard price trend for {}s".format(rarity))
        ax1.plot_date(std_dates, rarity_sum, '-', label=rarity, color=color_dict[rarity])
        standard_total = standard_total+rarity_sum
    print("Plotting total standard prices and set counts")   
    ax1.plot_date(std_dates, standard_total, '-', label='total', color='purple')
    ax2.plot_date(std_dates, std_sets.sum(), '-',color='g', label='# legal sets')

    # Format plot
    ax2.set_yticks(np.arange(0,11,1))
    month_formatter(ax1)
    month_formatter(ax2)
    ax1.set_xticks(std_dates)
    ax2.set_xticks(std_dates)
    ax1.grid(True)
    ax1.legend()
    ax2.legend()
    ax1.set_ylabel('Sum of Standard-legal card prices, in USD')
    ax1.set_xlabel('Standard Seasons')
    # ax1.set_title('')
    fig.suptitle('Total Price of Standard Cards', fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_all_standard_cards(rarities = ['mythic','rare', 'uncommon', 'common'],
                            alpha_dict = {'mythic':.1, 'rare':.1, 'uncommon':0.05, 'common':0.05},
                            color_dict = {'mythic':'r', 'rare':'goldenrod', 'uncommon':'grey', 'common':'cyan'},
                            log_price=True):
    # Define the standard format
    std_sets, std_dates = get_standard_format_real()
    
    # Plot setup
    fig, ax1 = plt.subplots()
    ax1.set_xticks(std_dates)

    # Plot every card in standard
    for rarity in rarities:
        print("Plotting all standard {} cards price history".format(rarity))
        standard_prices = get_standard_prices(rarity, std_sets)
        standard_prices.drop(columns=['cardname','setname'],inplace=True)
        for index, card in standard_prices.iterrows():
            ax1.plot_date(std_dates, card, '-', color=color_dict[rarity], label='_nolegend_', alpha=alpha_dict[rarity])
        print("Plotting mean standard {} price history".format(rarity))
        my_effects = [pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()]
        my_label = "avg "+rarity+ " $"
        ax1.plot_date(std_dates, standard_prices.mean(), '-', label=my_label, path_effects=my_effects, color=color_dict[rarity])

    if log_price:
        ax1.set_yscale("log", nonposy='clip')

    month_formatter([ax1])
    ax1.grid(True)
    ax1.legend()
    plt.show()    

def plot_all_cards(rarities = ['mythic','rare', 'uncommon', 'common'],
                   alpha_dict = {'mythic':.15, 'rare':0.05, 'uncommon':0.03, 'common':0.015},
                   color_dict = {'mythic':'r', 'rare':'goldenrod', 'uncommon':'grey', 'common':'cyan'},
                   log_price=True):
  
    # Get dates; generalize to work for any format; need seasonal averages not by format?
    dates_df = pd.read_csv('data/season_dates.csv')
    dates = pd.to_datetime(dates_df['end_date'].values)

    # Plot 
    fig, ax1 = plt.subplots()
    for rarity in rarities:
        print("Plotting all {} cards price history".format(rarity))
        seasonal_prices = pd.read_csv('data/clean_cards-{}_seasonal_avg.csv'.format(rarity))
        seasonal_prices.drop(columns=['cardname','setname'],inplace=True)
        for index, card in seasonal_prices.iterrows():
            ax1.plot(dates, card, color=color_dict[rarity], label='_nolegend_', alpha=alpha_dict[rarity])
        print("Plotting {} average price history".format(rarity))
        my_effects = [pe.Stroke(linewidth=3, foreground='k'), pe.Normal()]
        my_label = "avg "+rarity+ " $"
        ax1.plot(dates, seasonal_prices.mean(), label=my_label, path_effects=my_effects, color=color_dict[rarity])

    # Format plot
    if log_price:
        ax1.set_yscale("log", nonposy='clip')
    month_formatter([ax1])
    ax1.legend()
    plt.show()    

def clean_seasonal_price_outliers(rarities = ['mythic','rare', 'uncommon', 'common']):
    for rarity in rarities:
        # Load and index cards
        cards_df = pd.read_csv('data/all_vintage_cards-{}_seasonal_avg.csv'.format(rarity))
        cards_df.set_index(['cardname','setname'],inplace=True)

        # Drop collectibles
        basics = ['Plains','Island','Swamp','Mountain','Forest']
        sets_to_drop = [
            'Kaladesh Inventions',
            'Zendikar Expeditions',
            'Portal Three Kingdoms',
            'Legends',
            'Arabian Nights',
            'Modern Masters',
            'Eternal Masters',
            'Modern Masters 2017',
            'Modern Masters 2015',
            'Iconic Masters',
            'Portal Second Age',
            'Portal',
            'The Dark',
            'Battle Royale Box Set',
            'Beatdown Box Set',
            'Starter 2000',
            'Starter 1999',
            'Prerelease Events',
            'Release Events',
            # 'Commander 2013',
            # 'Commander 2014',
            # 'Commander 2015',
            # 'Commander 2016',
        ]
        cards_df.drop('Unnamed: 0',axis=1,inplace=True)
        cards_df.drop(sets_to_drop,level='setname',axis=0,inplace=True)
        cards_df.drop(basics,level='cardname',axis=0,inplace=True)

        # Reset index and rewrite
        cards_df.reset_index(inplace=True)
        cards_df.to_csv('data/clean_cards-{}_seasonal_avg.csv'.format(rarity),index=False)

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

# TODO: merge prices + feature data and write to file
def write_seasonal_averages(rarities):
    cards_df = pd.read_csv('data/all_vintage_cards.csv')
    seasons = np.array(pd.read_csv("data/season_dates.csv"))
    for rarity in rarities:
        tablename = rarity+"_price_history_2"
        print("Starting seasonal w-avg price query for {} cards".format(rarity))
        seasonal_price_history = w_avg_price_by_season(seasons, tablename)
        print("Writing seasonal {} prices to csv".format(rarity))
        seasonal_price_history.to_csv(path_or_buf='data/all_vintage_cards-{}_seasonal_avg.csv'.format(rarity))

def combine_csv_rarities(card_view='all_vintage_cards', price_type='recent', data='data'):
    rarities = ['mythic','rare','uncommon','common']
    dfs = []
    for rarity in rarities:
        filename = '{DATA}/{CARD_VIEW}-{RARITY}_{PRICE_TYPE}.csv'.format(DATA=data,
                                                                         CARD_VIEW=card_view,
                                                                         RARITY=rarity,
                                                                         PRICE_TYPE=price_type)
        try:
            dfs.append(pd.read_csv(filename))
        except FileNotFoundError:
            raise RuntimeError("Could not find {}. Make sure the path is correct.".format(filename))
    
    cards_df = pd.concat(dfs)
    return cards_df

def join_features_seasonal_prices():
    """ Joins card features with seasonal average prices """
    prices_df = combine_csv_rarities(card_view='all_vintage_cards', price_type='seasonal_avg')
    features_df = pd.read_csv('data/all_vintage_cards.csv',index_col=None)
    merged_df = pd.merge(features_df, prices_df, left_on=['name','set_name'], 
                                                 right_on=['cardname','setname'],
                                                 how='inner')
    merged_df.drop(columns=['name', 'set_name','Unnamed: 0'], inplace=True)
    merged_df.set_index('id', inplace=True)
    
    return merged_df

def get_craigs_picks():
    setname = "Ixalan"
    cards = [
                ["Carnage Tyrant", 'buy'],
                # ["Search for Azcanta // Azcanta, the Sunken Ruin",'buy'],
                ["Admiral Beckett Brass", 'buy'],
                ["Axis of Mortality", 'sell'],
                ["Boneyard Parley", 'sell'],
                ["Wakening Sun's Avatar", 'buy'],
                ["Jace, Cunning Castaway",'sell'],
                ["Overflowing Insight",'sell'],
                ["Dire Fleet Ravager",'sell'],
                ["Rowdy Crew",'buy'],
                ["Star of Extinction",'sell'],
                ["Vraska, Relic Seeker",'sell'],
                ["Huatli, Warrior Poet",'sell'],
                ["Vona, Butcher of Magan",'sell'],
                ["Gishath, Sun's Avatar",'sell'],
                ["Tishana, Voice of Thunder",'sell'],
                # ["Legion's Landing",'sell'],
                # ["Vance's Blasting Cannons",'sell'],
                # ["Growing Rites of Itlimoc",'sell'],
                ["Deathgorge Scavenger",'sell'],
                ["Bishop of Rebirth",'sell']
            ]
    
    all_cards_df = pd.read_csv('data/all_vintage_cards.csv', low_memory=False)
    picks_df = pd.DataFrame()
    for card in cards:
        cardname = card[0]
        print("trying {}".format(cardname))
        # find card in all cards data, get row
        mask = (all_cards_df['name']==cardname) & (all_cards_df['set_name']==setname)

        cardname = cardname.split('/')[0]

        cardrow = all_cards_df[mask].iloc[0]
        # get card price history 
        price_history = get_card_price_history(cardrow)
        # convert dates and plot 

        # store prices just before and after prediction into dataframe 
        if len(price_history):
            cardrow['before'] = price_history[0][1]
            cardrow['after'] = price_history[-1][1]
        cardrow['craig_pred'] = card[1]
        picks_df = picks_df.append(cardrow)

    return picks_df, cards




if __name__ == "__main__":
    # cardname= "Search for Azcanta // Azcanta, the Sunken Ruin"
    # get_card_price_history(cardname, "Ixalan")
    get_craigs_picks()