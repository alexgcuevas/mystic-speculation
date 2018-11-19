import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
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

class DeriveFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Add engineered features to DataFrame."""

    def fit(self, X, y=None):
        """Does not save state"""

        return self

    def transform(self, X):
        """Derives additional features used in the training of models."""

        df = derive_features(X)
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

def derive_features(X):
    df = X.copy()
 
    # Difficulty casting
    df['mana intensity'] = df['mana_cost'].apply(lambda x: len(x))
    df['color intensity'] = df['color_identities'].apply(lambda x: len(x))

class SelectFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Select features."""

    # TODO: add parameterization of features for code reuse (or find a generic transformer)
    def __init__(self):
        self.features_to_drop = [
            # FILL WITH FEATURES TO DROP; SEE SCRAPER?
        ]

    def fit(self, X, y=None):
        """Does not save state."""

        return self

    def transform(self, X):
        """'Return DataFrame containing only the configured features."""

        df = X.drop(self.features_to_drop, axis=1)

        return df

class ReadCSVTransformer(BaseEstimator, TransformerMixin):
    """Clean up after reading CSV"""

    def fit(self, X, y=None):
        """Does not save state."""
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()
        df.drop(columns='Unnamed: 0', inplace=True)
        df.drop_duplicates(inplace=True)
        df.set_index('id', inplace=True)
        return df

class SetExclusionTransformer(BaseEstimator, TransformerMixin):
    """Removes sets"""
    def __init__(self):
        self.Sets_to_drop = [
            'Kaladesh Inventions',
            'Zendikar Expeditions'
        ]
    def fit(self, X, y=None):
        """Does not save state."""
        return self

    def transform(self, X):
        """Drops unnamed column & duplicates, and sets id as index"""
        df = X.copy()
        df.drop(columns='Unnamed: 0', inplace=True)
        df.drop_duplicates(inplace=True)
        df.set_index('id', inplace=True)
        return df