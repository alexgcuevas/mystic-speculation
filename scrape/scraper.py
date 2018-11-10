import numpy as np
import pandas as pd
import json, requests, pickle, time
from bs4 import BeautifulSoup
# trying slimit parser
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor

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

def card_price_history(setname, cardname):
    # Turn card data into soup
    link = 'https://www.mtgprice.com/sets/' + '_'.join(setname.split()) + '/' + '_'.join(cardname.split())
    soup = BeautifulSoup(requests.get(link).content, 'html.parser')

    # GET RESULTS
    text_to_find = 'var results = ['
    history=[]
    for script in soup.findAll('script', type='text/javascript'):
        if text_to_find in script.text:
            parser = Parser()
            tree = parser.parse(script.text)
            for node in nodevisitor.visit(tree):
                if isinstance(node, ast.Assign) and getattr(node.left, 'value', '') == "\"data\"":
                    for prices in node.right.items:
                        history.append([prices.items[0].value,prices.items[1].value])
                    break
    return np.array(history)

def sets_price_history(sets, all_cards_df):
    set_dict = {}
    for setname in sets:
        print(setname)
        cards = all_cards_df[all_cards_df['set_name'] == setname]['name'].values
        card_dict = {}
        for cardname in cards:
            if '/' in cardname:
                cardname = cardname.split('/')[0]
            print(cardname)
            try:
                history = card_price_history(setname, cardname)
                card_dict[cardname] = history
            except:
                print('{} not a set on MTGPrice'.format(setname))
                break
        set_dict[setname] = card_dict
    return set_dict

def load_card_features(n=1320):
    # read n pages (1320 total as of 11/8/2018)
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

if __name__ == "__main__":
    all_cards_df = pd.read_csv('all_vintage_cards.csv')
    sets = list(all_cards_df['set_name'].unique())
    set_dict = sets_price_history(sets, all_cards_df)
    with open("price_scrape.p", 'wb') as output_file:
        pickle.dump(set_dict, output_file)