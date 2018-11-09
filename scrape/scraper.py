import numpy as np
import pandas as pd
import json, requests, pickle, time
from bs4 import BeautifulSoup

def load_card_page(page):
    '''
    Gets a page of scryfall cards from their API and formats it into a dataframe, removing unwanted cards.
    Input:
        page is a positive integer representing the page number
    Output:
        Returns a pandas dataframe of all cards on the page 
    '''
    link = 'https://api.scryfall.com/cards?page='+str(page)
    response = requests.get(link)
    cards = response.json()['data']
    cards_df = pd.DataFrame(cards)
    # Filters not legal in vintage (tokens, joke cards, conspiracies, etc.), only english cards
    legal_cards = cards_df[(cards_df['legalities'].apply(lambda x: x['vintage']!='not_legal')) & (cards_df['lang']=='en')]
    legal_cards.set_index('id', inplace=True)
    return legal_cards

def MVP_features(cards_df):
    '''
    Filters a dataframe of cards for only those cards and features to be included in
    MVP model.
    Input:
        Dataframe of cards, with card features as columns
    Output:
        Cleaned dataframe
    '''
    # Features to keep
    MVP_features = [
        'name',
        'set_name',
        'type_line',    
        'mana_cost',
        'rarity',
        'oracle_text',
        'power',
        'toughness',
        'loyalty',
        'cmc',
        'set',
        'color_identity',
        'colors',    
        'reprint',
        'layout',
        'legalities',
    ]
    misc_features = [
        'all_parts',
        'artist',
        'border_color',
        'card_faces',
        'edhrec_rank',
        'flavor_text',
        'foil',
        'nonfoil',
        'full_art',
        'watermark'    
        'timeshifted',
        'colorshifted',
        'futureshifted',
        'illustration_id',
        'multiverse_ids',
        'oracle_id',
        'prints_search_uri',
        'rulings_uri',
        'set_search_uri',
    ]

    # # Fills features with nan if not present in cards
    # clean_feats = set(clean_cards.columns)
    # extra_feats = set(MVP_features) - clean_feats
    # for feat in extra_feats:
    #     clean_cards[feat] = np.nan
    return cards_df[MVP_features]

if __name__ == "__main__":
    # read n pages (1320 total as of 11/8/2018)
    n = 1320
    cards = pd.DataFrame()
    for n in range(n):
        page = load_card_page(n+1)
        cards = pd.concat([cards, page], sort=True)
        print('just scraped this page: {}'.format(n+1))
        print('this many cards so far: {}'.format(cards.shape[0]))
        # sleep for 10 ms per scryfall API guidelines
        time.sleep(0.01)
    print('FINAL CARD TALLY: {}'.format(cards.shape[0]))
    print(' ~~~ cleaning everything now ~~~ ')
    MVP_data = MVP_features(cards)
    print(' ~~~ writing to csv ~~~ ')
    MVP_data.to_csv(path_or_buf='all_vintage_cards.csv')