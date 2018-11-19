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

def test_feature_engineering(cards_df):
    # Test creature feature eng
    # 1 normal creature, one weird creature, one spell
    creature = cards_df[cards_df['name']=='Arcanis the Omnipotent'].iloc[0]
    spell = cards_df[cards_df['name']=='Momentary Blink'].iloc[0]
    weird = cards_df[cards_df['name']=='Tarmogoyf'].iloc[0]
    tests = pd.concat([creature, spell, weird], axis=1).T

    creature_feature = CreatureFeatureTransformer()
    new_feats_df = creature_feature.transform(tests)
    new_feats = ['name']+list(set(new_feats_df.columns) - set(cards_df.columns))
    print(new_feats_df[new_feats])

def query_rarity_dfs(rarity='mythic', version=2, rows=10):
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
    results = connection.execute(query)
    recent_df = pd.read_sql(query, connection)
    connection.close()
    return recent_df.sample(rows)

if __name__ == "__main__":
    # run tessssts
    recent_df = query_rarity_dfs('rare')
    print(recent_df)


