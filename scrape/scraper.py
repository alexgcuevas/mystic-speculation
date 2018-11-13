import numpy as np
import pandas as pd
import json, requests, pickle, time
from bs4 import BeautifulSoup
# trying slimit parser
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
# connect to postgresql database
from sqlalchemy import create_engine

def MVP_features(cards_df):
    '''
    Filters a dataframe of cards for only those cards and features to be included in
    MVP model.
    Input:
        cards_df, dataframe of cards with card features as columns
    Output:
        dataframe containing only the desired features from the input cards
    '''
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
    return cards_df[MVP_features]

def load_card_page(page):
    '''
    Gets a page of scryfall cards from their API and formats it into a dataframe, removing unwanted cards.
    Input:
        page is a positive integer representing the page number
    Output:
        legal_cards, a pandas dataframe of all cards on the page 
    '''
    link = 'https://api.scryfall.com/cards?page='+str(page)
    response = requests.get(link)
    cards = response.json()['data']
    cards_df = pd.DataFrame(cards)
    
    # Filters not legal in vintage (tokens, joke cards, conspiracies, etc.), only english cards
    legal_cards = cards_df[(cards_df['legalities'].apply(lambda x: x['vintage']!='not_legal')) & (cards_df['lang']=='en')]
    legal_cards.set_index('id', inplace=True)
    return legal_cards

def load_card_features(n=1320):
    '''
    Loads cards from scryfall API, up to n pages (scryfall cards are paginated), and selecting
    only the desired card features.
    Input:
        n is number of scryfall pages to search - default is all of them.
    Output:
        None; writes the extracted cards to csv file. 
    '''
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

def card_price_history(setname, cardname):
    '''
    Scrapes price history of card from MTGPrice.com, using javascript parser
    Input:
        Setname and cardname are strings, generally taken from Scryfall API.
    Output:
        A numpy array of price history, each 'row' in the form [timestamp, price]
    '''
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
    '''
    Scrapes price data from MTGPrice.com for all cards in a given list of sets.
    Input:
        sets is a list of sets to scrape, all_cards_df is a pandas dataframe of cards
    Output:
        set_dict is a dictionary with keys being magic set names from 'sets', and values
        being dictionaries of (cards, price history) kv pairs for each card in the set.
    '''
    set_dict = {}
    for setname in sets:
        print('Scraping set from MTGPrice.com: {}'.format(setname))
        cards = all_cards_df[all_cards_df['set_name'] == setname]['name'].values
        card_dict = {}
        for i, cardname in enumerate(cards):
            if '/' in cardname:
                cardname = cardname.split('/')[0]
            print('Scraping card from MTGPrice.com: {}'.format(cardname))
            try:
                history = card_price_history(setname, cardname)
                card_dict[cardname] = history
                print('successfully scraped {0} from {1}'.format(cardname, setname))
                time.sleep()
            except:
                if i == 1:
                    print('SET SCRAPE FAIL!\nfailed set: {}'.format(setname))
                    break
                else:
                    print('CARD SCRAPE FAIL!\nfailed at #{0} card: {1}'.format(i, cardname))                
        set_dict[setname] = card_dict
    return set_dict

def load_to_postgres(cardname, setname, price_dict):
    price_df = pd.DataFrame(price_dict)
    engine = create_engine('postgresql://mystic-speculation.cwxojtlggspu.us-east-1.rds.amazonaws.com:password?@localhost:5432/alexgcuevas')
    price_df.to_sql('sets', engine)
    pass

if __name__ == "__main__":
    all_cards_df = pd.read_csv('all_vintage_cards.csv')
    sets = list(all_cards_df['set_name'].unique())
    set_dict = sets_price_history(sets, all_cards_df)
    with open("all_vintage_price_scrape.p", 'wb') as output_file:
        pickle.dump(set_dict, output_file)