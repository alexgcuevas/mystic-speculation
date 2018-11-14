import numpy as np
import pandas as pd

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

independent / dependent transforms:
    card-card similarity, nearest neighbors
    price normalization by season
    price decomposition by format
'''
