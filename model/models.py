from model.master_transmuter import *
from scrape.scraper import *
from query import *
import unit_tests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def format_results(X_test, y_pred, y_test):
    results_df = X_test.copy()
    results_df['y_pred'] = y_pred
    results_df['y_test'] = y_test
    results_df['log_diff'] = np.abs(np.log(y_pred+1) - np.log(y_test+1))
    return results_df

def baseline_model(X_train, X_test, y_train, y_test):
    """
        DEPRECATED
        Guesses mean price by rarity
    """

    # The set-up (you need this)
    y_train_log = np.log(y_train)
    y_pred_log = np.ones(y_test.shape)

    # Fit & Predict
    for rarity in X_train['rarity'].unique():
        train_mask = X_train['rarity']==rarity
        avg = y_train_log[train_mask].mean()
        test_mask = X_test['rarity']==rarity
        y_pred_log[test_mask] = avg

    y_pred = price_corrector(np.exp(y_pred_log))

    results_df = format_results(X_test, y_pred, y_test)
    score = -rmlse(y_pred, y_test)

    return results_df, score

def fit_basic_pipeline(X_train, X_test, y_train, y_test):

    # The set-up (you need this)
    pipe = Pipeline([
        ('CreatureFeature', CreatureFeatureTransformer()),
        ('Planeswalker', PlaneswalkerTransformer()),
        ('BoolToInt', BoolTransformer()),
        ('Fillna', FillTransformer()),
        ('CostIntensity', CostIntensityTransformer()),
        ('DropFeatures', DropFeaturesTransformer()),
        ('CreateDummies', CreateDummiesTransformer()),
        ('TestFill', TestFillTransformer()),
        ('GradientBoostingRegressor', GradientBoostingRegressor())
    ])

    y_train_log = np.log(y_train)

    # Fit & Predict
    pipe.fit(X_train, y_train_log)
    y_pred_log = pipe.predict(X_test)
    y_pred = price_corrector(np.exp(y_pred_log))

    results_df = format_results(X_test, y_pred, y_test)
    score = -rmlse(y_pred, y_test)

    return pipe, results_df, score

def plot_residuals(y_pred, y_test, title):
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    lin_ax = axs[0]
    log_ax = axs[1]

    lin_ax.scatter(y_test, y_pred, label='preds', alpha=0.5)
    lin_ax.scatter(y_test, y_test, label='actuals', alpha=0.5)
    lin_ax.set_title('{} predictions vs actual'.format(title))
    lin_ax.set_ylabel('predicted prices')
    lin_ax.set_xlabel('actual prices')
    lin_ax.set_xlim(0,max(y_test.max(), y_pred.max()))
    lin_ax.set_ylim(0,max(y_test.max(), y_pred.max()))

    log_ax.scatter(np.log(y_test+1), np.log(y_pred+1), label='log preds', alpha=0.5)
    log_ax.scatter(np.log(y_test+1), np.log(y_test+1), label='log actuals', alpha=0.5)
    log_ax.set_title('{} log predictions vs actual'.format(title))
    log_ax.set_ylabel('predicted log prices')
    log_ax.set_xlabel('actual log prices')
    logs = np.log(y_test.append(y_pred)+1)
    log_ax.set_xlim(logs.min(),logs.max())
    log_ax.set_ylim(logs.min(),logs.max())

    plt.legend()
    plt.show()

def plot_residuals_vs_baseline(results_df, baseline_df, title):
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    lin_ax = axs[0]
    log_ax = axs[1]

    y_test = results_df['y_test']
    y_pred = results_df['y_pred']
    y_base = baseline_df['y_pred']

    lin_ax.scatter(y_test, y_pred, label='preds', alpha=0.5)
    lin_ax.scatter(y_test, y_test, label='actuals', alpha=0.5)
    lin_ax.scatter(y_test, y_base, label='baseline', alpha=0.5)    
    lin_ax.set_title('{} predictions vs actual'.format(title))
    lin_ax.set_ylabel('predicted prices')
    lin_ax.set_xlabel('actual prices')
    lin_ax.set_xlim(0,max(y_test.max(), y_pred.max()))
    lin_ax.set_ylim(0,max(y_test.max(), y_pred.max()))

    log_ax.scatter(np.log(y_test+1), np.log(y_pred+1), label='log preds', alpha=0.5)
    log_ax.scatter(np.log(y_test+1), np.log(y_test+1), label='log actuals', alpha=0.5)
    log_ax.scatter(np.log(y_test+1), np.log(y_base+1), label='baseline', alpha=0.5)        
    log_ax.set_title('{} log predictions vs actual'.format(title))
    log_ax.set_ylabel('predicted log prices')
    log_ax.set_xlabel('actual log prices')
    logs = np.log(y_test.append(y_pred)+1)
    log_ax.set_xlim(logs.min(),logs.max())
    log_ax.set_ylim(logs.min(),logs.max())

    lin_ax.legend()
    log_ax.legend()
    plt.show()

def plot_pred_hist(y_pred, y_test, title):
    plt.figure(figsize=(10,10))
    plt.hist(np.array([y_test,y_pred]).T, label=['actuals','preds'], bins=30)
    plt.title('{} predictions vs actual, histogram'.format(title))
    plt.ylabel('number predicted')
    plt.xlabel('prices')
    plt.legend()
    plt.show()

def rmlse(y_pred,y_test):
    """ Calculates Room Mean Log Squared Error of prediction """
    log_diff = np.log(y_pred+1) - np.log(y_test+1)
    return np.sqrt(np.mean(log_diff**2))

def rmlse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    log_diff = np.log(y_pred+1) - np.log(y+1)
    return -np.sqrt(np.mean(log_diff**2))

def price_corrector(y_pred):
    y_pred = np.array(y_pred) 
    y_pred[y_pred < 0.10] = 0.10
    return np.round(y_pred,1)

def fit_refine_pipeline(X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ('BoolToInt', BoolTransformer()),
        ('CreatureFeature', CreatureFeatureTransformer()),
        ('Planeswalker', PlaneswalkerTransformer()),
        ('AbilityCounts', AbilityCountsTransformer()),
        ('Fillna', FillTransformer()),
        ('CostIntensity', CostIntensityTransformer()),
        ('CreateDummies', CreateDummiesTransformer()),
        ('DummifyType', TypelineTransformer()),
        ('DummifyColorID', ColorIDTransformer()),
        ('DropFeatures', DropFeaturesTransformer()),
        ('TestFill', TestFillTransformer()),
        ('GradientBoostingRegressor', GradientBoostingRegressor())
    ])

    y_train_log = np.log(y_train)
    # y_test_log = np.log(y_test)

    pipe.fit(X_train, y_train_log)
    y_pred_log = pipe.predict(X_test)
    y_pred = price_corrector(np.exp(y_pred_log))

    results_df = X_test.copy()
    results_df['y_pred'] = y_pred
    results_df['y_test'] = y_test
    results_df['log_diff'] = np.abs(np.log(y_pred+1) - np.log(y_test+1))
    score = -rmlse(y_pred, y_test)

    return pipe, results_df, score

def pipe_feature_imports(pipe):
    """
    Takes in pipeline with decision tree as estimator,
    returns dataframe of feature importances
    """
    model = pipe.steps[-1][1]
    features = list(pipe.steps[-2][1].train_columns)
    feature_importances = np.round(model.feature_importances_,4)

    feature_importances = np.array([features, feature_importances]).T
    return pd.DataFrame(feature_importances[feature_importances[:,1].argsort()[::-1]], columns=['feature','importance'])

# TODO NEED TO WRITE PROPER MODELS
def run_models_against_baseline(models, cards_df, scorer, n_folds=5):
    """
        Assumes cards_df has 'price' as y column 
        models is list of [model, modelname]
    """
    # Formats csv into df, excludes sets, and  
    X, y = csv_cleaner(cards_df)
    
    # Cross-validate model & predict
    score_dict = {}

    baseline = BaselineModel()
    base_scores = cross_val_score(baseline, X, y, cv=n_folds, verbose=2, scoring=scorer, n_jobs=-1)

    for model, modelname in models:
        my_scores = cross_val_score(model, X, y, cv=n_folds, verbose=2, scoring=scorer, n_jobs=-1)
        score_dict[modelname] = my_scores

    print("baseline scores: \n\t{}".format(base_scores))
    print("baseline average: \n\t{}".format(np.mean(base_scores)))

    for modelname, scores in score_dict.items():
        print("{MODEL} scores: \n\t{SCORES}".format(MODEL=modelname, SCORES=scores))
        print("{MODEL} average: \n\t{SCORES}".format(MODEL=modelname, SCORES=np.mean(scores)))

    # Format & plot results
    return score_dict

def model_gauntlet():
    cards_df = combine_csv_rarities()
    scorer = rmlse_scorer

    model_a = SpotPriceGBR()
    modelname_a = "SpotPriceGBR"
    pipe_a = create_pipeline(model_a, modelname_a)

    model_a1 = SpotPriceGBR(log_y=True)
    modelname_a1 = "SpotPriceGBR_log"
    pipe_a1 = create_pipeline(model_a1, modelname_a1)

    model_b = SpotPriceByRarityGBR()
    modelname_b = "SpotPriceByRarityGBR"
    pipe_b = create_pipeline(model_b, modelname_b)
    
    model_b1 = SpotPriceByRarityGBR(log_y=True)
    modelname_b1 = "SpotPriceByRarityGBR_log"
    pipe_b1 = create_pipeline(model_b, modelname_b)

    run_models_against_baseline([[pipe_a, modelname_a],
                                 [pipe_a1, modelname_a1],
                                 [pipe_b, modelname_b],
                                 [pipe_b1, modelname_b1]], 
                                 cards_df, scorer, n_folds=10)

def GBR_V1():
    pipe = Pipeline([
        ('BoolToInt', BoolTransformer()),
        ('CreatureFeature', CreatureFeatureTransformer()),
        ('Planeswalker', PlaneswalkerTransformer()),
        ('AbilityCounts', AbilityCountsTransformer()),
        ('Fillna', FillTransformer()),
        ('CostIntensity', CostIntensityTransformer()),
        ('CreateDummies', CreateDummiesTransformer()),
        ('DummifyType', TypelineTransformer()),
        ('DummifyColorID', ColorIDTransformer()),
        ('DropFeatures', DropFeaturesTransformer()),
        ('TestFill', TestFillTransformer()),
        ('SpotPriceGBR', SpotPriceGBR())
    ])
    return pipe

def create_pipeline(model, modelname):
    pipe = Pipeline([
        ('BoolToInt', BoolTransformer()),
        ('CreatureFeature', CreatureFeatureTransformer()),
        ('Planeswalker', PlaneswalkerTransformer()),
        ('AbilityCounts', AbilityCountsTransformer()),
        ('Fillna', FillTransformer()),
        ('CostIntensity', CostIntensityTransformer()),
        ('CreateDummies', CreateDummiesTransformer()),
        ('DummifyType', TypelineTransformer()),
        ('DummifyColorID', ColorIDTransformer()),
        ('DropFeatures', DropFeaturesTransformer()),
        ('TestFill', TestFillTransformer()),
        (modelname, model)
    ])
    return pipe

class BaselineModel(BaseEstimator, RegressorMixin):
    """Baseline Model to evaluate mine against"""
    def __init__(self):
        self.rarity_averages_ = {}

    def fit(self, X, y):
        # Store average prices by rarity for training data
        for rarity in X['rarity'].unique():
            train_mask = X['rarity']==rarity
            self.rarity_averages_[rarity] = y[train_mask].mean()
        
        return self

    def predict(self, X):
        """ Predict mean prices by rarity, based on training data """
        # Make sure the model has been fit with averages
        try:
            getattr(self, "rarity_averages_")
        except AttributeError:
            raise RuntimeError("rarity_averages_ doesn't exist; make sure you fit a model first")    

        y_pred = np.ones(X.shape[0])

        for rarity in X['rarity'].unique():
            test_mask = X['rarity']==rarity
            if rarity in self.rarity_averages_.keys():
                y_pred[test_mask] = self.rarity_averages_[rarity]
            else:
                y_pred[test_mask] = self.rarity_averages_.values().next()
        
        return y_pred

    def score(self, X, y):
        """ Use RMLSE; Root Mean Log Squared Error. Score is -RMLSE, because bigger is better"""
        y_pred = self.predict(X)
        return -rmlse(y_pred, y)

# TODO custom scorer, predict floor
class SpotPriceGBR(BaseEstimator, RegressorMixin):
    """ Model using only recent prices (done already; need to formalize)"""
    def __init__(self, model=GradientBoostingRegressor(), base_weight=0, log_y=False):
        self.base_weight = base_weight
        self.model = model
        self.log_y = log_y

    def fit(self, X_train, y_train):
        X = X_train.copy()
        y = y_train.copy()

        if self.log_y:
            y = np.log(y)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """ Set floor to GBR predictions """
        y_pred = self.model.predict(X)

        if self.log_y:
            y_pred = np.exp(y_pred)

        y_pred[y_pred<0.1]=0.1
        return y_pred

    def score(self, X, y):
        """ Use RMLSE; Root Mean Log Squared Error. Score is -RMLSE, because bigger is better"""
        y_pred = self.predict(X)
        return -rmlse(y_pred, y)

class SpotPriceByRarityGBR(BaseEstimator, RegressorMixin):
    """ Model using only recent prices, fitting models by rarity"""
    def __init__(self, model=GradientBoostingRegressor(), base_weight=0,
                 log_y=False, rarities=['mythic', 'rare', 'uncommon', 'common']):
        self.base_weight = base_weight
        self.model = model
        self.log_y = log_y
        self.rarities = rarities
        self.rarity_models_ = {}

    def fit(self, X_train, y_train):
        X = X_train.copy()
        y = y_train.copy()

        if self.log_y:
            y = np.log(y)
        
        self.train_rarities_ = [x for x in X.columns if x.startswith('rarity_')]

        for rarity in self.train_rarities_:
            train_mask = X[rarity]==1
            rarity_model = clone(self.model)
            rarity_model.fit(X[train_mask], y[train_mask])
            self.rarity_models_[rarity] = rarity_model
        
        return self

    def predict(self, X):
        """ Set floor to GBR predictions """
        try:
            getattr(self, "rarity_models_")
        except AttributeError:
            raise RuntimeError("rarity_models_ doesn't exist; make sure you fit a model first")    

        y_pred = np.ones(X.shape[0])

        test_rarities = [x for x in X.columns if x.startswith('rarity_')]

        for rarity in test_rarities:
            test_mask = X[rarity]==1
            
            if (rarity in self.train_rarities_) and (rarity in self.rarity_models_.keys()):    
                y_pred[test_mask] = self.rarity_models_[rarity].predict(X[test_mask])
            else:
                y_pred[test_mask] = self.rarity_models_.values().next().predict(X[test_mask])
        
        if self.log_y:
            y_pred = np.exp(y_pred)

        y_pred[y_pred<0.1]=0.1
        return y_pred

    def score(self, X, y):
        """ Use RMLSE; Root Mean Log Squared Error. Score is -RMLSE, because bigger is better"""
        y_pred = self.predict(X)
        return -rmlse(y_pred, y)

class StandardNormalizerModel(BaseEstimator, RegressorMixin):
    """Model using only recent prices (done already; need to formalize"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Cleave split cards and transforms
        return self

    def transform(self, X):
        df = X.copy()
        return df