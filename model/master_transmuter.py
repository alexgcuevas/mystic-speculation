import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# NLP Pipeline
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Silence outdated numpy warning
import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


'''
Features to engineer:
    pure feature transforms:
        p/t ratio
        t/p ratio
        p+t : cmc ratio
    nlp / text analysis:
        keywordification
        n-grams (the big boi)
            n-grams by card type?
        # abilities (count ':'s?)
        cmc of abilities
        synergies (tf-df?)
        mana abilities
        difficulty to cast (mana, 'additional cost')
        card similarity
        activated vs triggered abilities 
    independent / dependent transforms:
        card-card similarity, nearest neighbors
        price normalization by season
        price decomposition by format
'''
def one_hot(input_df, columns):
    """
    One-hot encode the provided list of columns and return a new copy of the data frame
    """
    df = input_df.copy()

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        df = df.drop(col, axis=1).merge(dummies, left_index=True, right_index=True)

    return df

class OneHotTransformer(BaseEstimator, TransformerMixin):
    """
    One-hot encode features
    """

    def fit(self, X, y=None):
        """
        Store the features resulting from training features
        Accepts DataFrame
        Saves state and returns self
        """
        df = one_hot(
            X,
            [
                # PUT FEATURES TO ONE-HOT HERE
            ],
        )
        self.train_columns = df.columns

        return self

    def transform(self, X):
        """
        One-hot encode and ensure all features captured in training are present as well.
        Accepts DataFrame
        Returns DataFrame with addition features
        """
        df = X.copy()
        df = one_hot(
            df,
            [
                # PUT FEATURES TO ONE-HOT HERE
            ],
        )

        # Remove untrained columns
        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0

        # Add trained on columns
        for col in df.columns:
            if col not in self.train_columns:
                df.drop(col, axis=1, inplace=True)

        return df[self.train_columns]

class FillTransformer(BaseEstimator, TransformerMixin):
    """
    Impute NaN values
    # TODO: Parameterize so values can be imputed with -1, mean, median, or mode.
    """

    def fit(self, X, y=None):
        self.fill_value = -1 # >>> DO I WANT THIS? 
        return self

    def transform(self, X):
        # paramaterize this with mean, median, mode, etc.
        # fill with -1
        # TODO: make this fill dynamic for all columns?
        df = X.copy()
        df.fillna(self.fill_value, axis=1, inplace=True)
        return df

class CostIntensityTransformer(BaseEstimator, TransformerMixin):
    """Add engineered features to DataFrame."""

    def fit(self, X, y=None):
        """Does not save state"""
        return self

    def transform(self, X):
        """Derives additional features used in the training of models."""
        df = X.copy()

        def colors(row):
            try:
                intensity = len(row['mana_cost'])
                if (intensity == 3) & (row['mana_cost'][1] not in ['W', 'U', 'B', 'R', 'G']):
                    return 0
                else: 
                    return intensity
            except:
                return 0
        # Difficulty casting
        df['mana_intensity'] = df['mana_cost'].apply(lambda x: colors(x))-2
        df['color_intensity'] = df['color_identity'].apply(lambda x: len(x)-2)
        return df

class CreatureFeatureTransformer(BaseEstimator, TransformerMixin):
    """Add engineered creature features to DataFrame."""

    def fit(self, X, y=None):
        """Does not save state"""

        return self

    def transform(self, X):
        """Derives additional features used in the training of models."""
        df = X.copy()

        # Creature Features
        def pt_type(row):
            if (type(row['power']) == type('str')) and (type(row['toughness']) == type('str')):
                if '*' in row['power']+row['toughness'] or row['toughness']<='0':
                    return 'variable'
                return 'static'
            return 'none'
        def power_to_int(row):
            if row['pt_type']=='static': 
                return int(row['power'])
            else:
                return 0
        def tough_to_int(row):
            if row['pt_type']=='static':
                return int(row['toughness'])
            else:
                return 0

        # Create pt_type feature, convert static pts to ints
        df['pt_type'] = df.apply(pt_type, axis=1)
        df['power'] = df.apply(power_to_int, axis=1)
        df['toughness'] = df.apply(tough_to_int, axis=1)

        # Only engineer creatures with static PT
        mask = df['pt_type']=='static'

        # ACTUAL ENGINEERING
        df['p:t'] = df[mask]['power']/df[mask]['toughness']
        df['avg_pt'] = (df[mask]['power']+df[mask]['toughness'])/2
        df['cmc:apt'] = df[mask]['cmc']/df[mask]['avg_pt']

        return df

class PlaneswalkerTransformer(BaseEstimator, TransformerMixin):
    """Add engineered creature features to DataFrame."""

    def fit(self, X, y=None):
        """Does not save state"""

        return self

    def transform(self, X):
        """Derives additional features used in the training of models."""
        df = X.copy()
        def loyal_type(row):
            try:
                loyal = int(row['loyalty'])
                row['loyalty']=loyal
                row['l_type']='static'
                return row
            except:
                row['loyalty']=0
                row['l_type']='variable'
                return row

        df = df.apply(loyal_type, axis=1)
        
        return df

class DropFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Select features."""

    # TODO: add parameterization of features for code reuse (or find a generic transformer)
    def __init__(self):
        self.features_to_drop = [
            'cardname'
            ,'setname'
            ,'type_line'
            ,'mana_cost'
            ,'oracle_text'
            ,'set'
            ,'colors'
            ,'color_identity'
            ,'legalities'
            ,'timestamp'
            ,'card_types'
            ,'mod_types'
            ,'sub_types'
            # ,'price'
        ]

    def fit(self, X, y=None):
        """Does not save state."""
        return self

    def transform(self, X):
        """'Return DataFrame containing only the configured features."""
        df = X.drop(self.features_to_drop, axis=1)
        return df

def csv_cleaner(df, y_col='price'):
    clean_df = df.copy()
    # clean_df.drop(columns=['Unnamed: 0'], inplace=True)
    clean_df.drop_duplicates(inplace=True)
    # clean_df.set_index('id', inplace=True)

    set_excluder = SetExclusionTransformer()
    clean_df = set_excluder.transform(clean_df)
    return clean_df.drop(columns=y_col, axis=1), clean_df[y_col]

class SetExclusionTransformer(BaseEstimator, TransformerMixin):
    """Removes sets"""
    def __init__(self, sets_to_drop=[]):
        if sets_to_drop:
            self.sets_to_drop = sets_to_drop
        else:
            self.sets_to_drop = [
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
                'Rivals of Ixalan',
                                
                # 'Commander 2013',
                # 'Commander 2014',
                # 'Commander 2015',
                # 'Commander 2016',
            ]
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()
        return df[df['setname'].apply(lambda x: x not in self.sets_to_drop)]

class BoolTransformer(BaseEstimator, TransformerMixin):
    """Changes all Falses to 0 and Trues to 1"""
    def fit(self, X, y=None):
        """Does not save state."""
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()
        df['reprint'] = 1*df['reprint']
        return df

class CreateDummiesTransformer(BaseEstimator, TransformerMixin):
    """Creates Dummies for given features"""
    def __init__(self, dummy_features=['rarity','layout','pt_type','l_type']):
        self.dummy_features = dummy_features

    def fit(self, X, y=None):
        """Does not save state."""
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()
        df = pd.get_dummies(df, columns=self.dummy_features, prefix=self.dummy_features)
        return df

class TestFillTransformer(BaseEstimator, TransformerMixin):
    """Fills unmatched columns in X_test with -1"""
    def __init__(self):
        self.fill_value = -1

    def fit(self, X, y=None):
        """Does not save state."""
        self.train_columns = set(X.columns)
        return self

    def transform(self, X):
        """Enlarges test columns to equal train size"""
        df = X.copy()
        # Fill columns in train but not test with fill value
        missing = self.train_columns - set(X.columns)
        if missing:
            for column in missing:
                df[column] = self.fill_value
        # drop columns in test but not in train
        df = df[list(self.train_columns)]
        return df

class TypelineTransformer(BaseEstimator, TransformerMixin):
    """Creates Dummies for typeline"""
    def __init__(self):
        self.card_types = set(['Creature','Land','Instant','Sorcery','Enchantment','Artifact','Planeswalker'])
        self.sub_types = set()
        self.mod_types = set()

    def fit(self, X, y=None):
        """identifies all subtypes"""
        # Cleave split cards and transforms
        cards = [x.split('//') for x in X['type_line'].unique()]
        for card in cards:
            for subcard in card:
                types = subcard.split(' — ')
                self.mod_types.update(set(types[0].split()) - self.card_types)
                try:
                    self.sub_types.update(set(types[1].split()))
                except:
                    pass
                    
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()

        def type_sets(row):
            card_types = set()
            sub_types = set()
            mod_types = set()
            card = row['type_line'].split('//')
            for subcard in card:
                types = subcard.split(' — ')
                card_types.update(set(types[0].split()) & self.card_types)
                mod_types.update(set(types[0].split()) - self.card_types)
                try:
                    sub_types.update(set(types[1].split()))
                except:
                    pass
            row['card_types'] = card_types
            row['mod_types'] = mod_types
            row['sub_types'] = sub_types
            return row

        def type_dummies(row):
            for card_type in self.card_types:
                row[card_type] = 1*(card_type in row['card_types'])
            for mod_type in self.mod_types:
                row[mod_type] = 1*(mod_type in row['mod_types'])
            return row

        df = df.apply(type_sets, axis=1)
        # Dummify card type membership, type_mod membership
        df = df.apply(type_dummies, axis=1)

        return df

class ColorIDTransformer(BaseEstimator, TransformerMixin):
    """Creates Dummies for color identity"""
    def __init__(self):
        self.colors = set(['W','U','B','R','G'])

    def fit(self, X, y=None):
        """No saved state"""
        return self

    def transform(self, X):
        """Dummifies color identity for each card"""
        df = X.copy()

        def color_dummies(row):
            for color in self.colors:
                row[color] = 1*(color in row['color_identity'])
            return row

        # Dummify color identity membership
        df = df.apply(color_dummies, axis=1)

        return df

class AbilityCountsTransformer(BaseEstimator, TransformerMixin):
    """Creates counts for various ability types"""

    def fit(self, X, y=None):
        """No saved state"""
        return self

    def transform(self, X):
        """Reads text and counts ability blocks, activated, and triggered abilities"""
        df = X.copy()

        def count_abilities(row):
            txt = row['oracle_text']
            if pd.isnull(txt):
                txt = ''
                row['ability_sects'] = len(txt.split('\r\r\n'))
            else:
                row['ability_sects'] = len(txt.split('\r\r\n'))

            # Count ability blocks
            row['ability_sects'] = len(txt.split('\r\r\n'))
            # Count activated
            row['activated'] = txt.count(':')            
            # Count triggered
            row['triggered'] = len(re.findall('When|At|As',txt))
            # Count mana abilities
            row['mana_abilities'] = len(re.findall('Add|add', txt))
            
            return row

        # Dummify color identity membership
        df = df.apply(count_abilities, axis=1)

        return df

# TODO: This doesn't do anything right now
class SeasonNormalizerTransformer(BaseEstimator, TransformerMixin):
    """ Transforms seasonal price history into target """
    def __init__(self, seasons=[]):
        self.seasons = seasons

    def fit(self, X, y=None):
        """identifies all subtypes"""
        # Cleave split cards and transforms
        cards = [x.split('//') for x in X['type_line'].unique()]
        for card in cards:
            for subcard in card:
                types = subcard.split(' — ')
                self.mod_types.update(set(types[0].split()) - self.card_types)
                try:
                    self.sub_types.update(set(types[1].split()))
                except:
                    pass
                    
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()

        def type_sets(row):
            card_types = set()
            sub_types = set()
            mod_types = set()
            card = row['type_line'].split('//')
            for subcard in card:
                types = subcard.split(' — ')
                card_types.update(set(types[0].split()) & self.card_types)
                mod_types.update(set(types[0].split()) - self.card_types)
                try:
                    sub_types.update(set(types[1].split()))
                except:
                    pass
            row['card_types'] = card_types
            row['mod_types'] = mod_types
            row['sub_types'] = sub_types
            return row

        def type_dummies(row):
            for card_type in self.card_types:
                row[card_type] = 1*(card_type in row['card_types'])
            for mod_type in self.mod_types:
                row[mod_type] = 1*(mod_type in row['mod_types'])
            return row

        df = df.apply(type_sets, axis=1)
        # Dummify card type membership, type_mod membership
        df = df.apply(type_dummies, axis=1)

        return df

class StandardSeasonTransformer(BaseEstimator, TransformerMixin):
    """ Add features to season matrix """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """ X has 'set_count' and 'season' as column """
        Xt = X.copy()
        Xt['set_count_sq'] = np.square(Xt['set_count'])
        Xt['set_count_rt'] = np.sqrt(Xt['set_count'])
        Xt['sin_set'] = np.sin(Xt['set_count'])

        return Xt

class PriceToPowerTransformer(BaseEstimator, TransformerMixin):
    """ Transforms price to power by scaling according to rarity, based on observed trends """
    def __init__(self, rarity_baseline={'mythic':10,'rare':1.5,'uncommon':0.5,'common':0.2}):
        self.rarity_baseline = rarity_baseline

    def _get_seasons(self, df):
        """ finds seasons in columns of df and returns list of them """
        seasons = [x for x in df.columns if x.strip('s').isnumeric()] 
        return seasons

    def fit(self, X, y=None):
        """ calculates average price by rarity of cards, averages over seasons, loads into attribute. Requires 'rarity' column """
        self.rarity_scaler_ = self.rarity_baseline.copy()
        seasonal_prices_df = X.copy()
        seasons = self._get_seasons(seasonal_prices_df)

        for rarity in self.rarity_baseline.keys():
            rare_mask = seasonal_prices_df['rarity']==rarity
            if rare_mask:
                rare_mean = seasonal_prices_df[rare_mask][seasons].mean()
                for avg_price in rare_mean:
                    self.rarity_scaler_[rarity] = np.mean(self.rarity_scaler_[rarity], avg_price) 

        return self

    def transform(self, X, y_price):
        """ transforms price in dollars into unitless power metric (pegged to avg mythic price) """
        rarity_df = X.copy()
        y_power = np.ones(y_price.shape[0])

        for rarity in self.rarity_baseline.keys():
            rare_mask = rarity_df['rarity']==rarity
            if rare_mask:
                # Scales to mean mythic rare price
                ratio = self.rarity_scaler_['mythic']/self.rarity_scaler_[rarity]
                y_power[rare_mask] = ratio*y_price[rare_mask]

        return y_power

    def inverse_transform(self, X, y_power):
        """ transforms power back into price """
        rarity_df = X.copy()
        y_price = np.ones(y_power.shape[0])

        for rarity in self.rarity_baseline.keys():
            rare_mask = rarity_df['rarity']==rarity
            if rare_mask:
                ratio = self.rarity_scaler_[rarity]/self.rarity_scaler_['mythic']
                y_price[rare_mask] = ratio*y_power[rare_mask]

        return y_price


    # TODO optimize efficiency by combining code from fit and transform
    def fit_transform(self, X, y_price):
        """ Performs fit and transform in one step, returning transformed price to power """
        self.fit(X)
        y_power = self.transform(X, y_price)

        return y_power

class StandardPriceTransformer(BaseEstimator, TransformerMixin):
    """ Performs standard price masking on input df """
    def __init__(self, std_sets_df):
        """ std_sets_df is a dataframe with sets as the indices and seasons as the colums """
        self.std_sets_df = std_sets_df
        self.seasons_ = std_sets_df.columns
    
    def fit(self, X, y=None):
        return self

    def _standard_mask(self, row):
        for season in self.seasons_:
            row[season] = row[season]*self.std_sets_df.loc(row['setname'])[season]
        return row

    def transform(self, X, y=None):
        """ X is seasonal prices, to be filtered for standard only """
        seasonal_prices_df = X.copy()
        return seasonal_prices_df.apply(self._standard_mask, axis=1)