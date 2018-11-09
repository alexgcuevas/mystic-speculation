import numpy as np
import pandas as pd
import json, requests, pickle
from bs4 import BeautifulSoup

def load_card_page(page):
    '''
    Gets a page of scryfall cards from their API and formats it into a dataframe.
    Input:
        page is a positive integer representing the page number
    Output:
        Returns a pandas dataframe of all cards on the page 
    '''
    link = 'https://api.scryfall.com/cards?page='+str(page)
    response = requests.get(link)
    cards = response.json()['data']
    cards_df = pd.DataFrame(cards)
    return cards_df

def clean_cards_MVP(cards_df):
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
    
    # Filters: not legal in vintage (tokens, joke cards, conspiracies, etc.), only english cards
    clean_cards = cards_df[(cards_df['legalities'].apply(lambda x: x['vintage']!='not_legal')) & (cards_df['lang']=='en')]
    clean_cards.set_index('id', inplace=True)
    return clean_cards[MVP_features]

if __name__ == "__main__":
    page = load_card_page(1)
    clean = clean_cards_MVP(page)
    print(clean.head(5))
