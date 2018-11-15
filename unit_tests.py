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
    print(new_feats_df.head())



