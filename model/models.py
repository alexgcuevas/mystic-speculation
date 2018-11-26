from model.master_transmuter import *
from scrape.scraper import *
from query import *
import unit_tests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
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
    """guesses mean price by rarity only"""

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
    score = log_score(y_pred, y_test)

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
    log_diff = np.log(y_pred+1) - np.log(y_test+1)
    return np.sqrt(np.mean(log_diff**2))

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
    score = log_score(y_pred, y_test)

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
def run_model_against_baseline(model, cards_df, log_y=True, n_folds=5):
    """
        Assumes cards_df has 'price' as y column 
    """
    # Formats csv into df, excludes sets, and  
    X, y = csv_cleaner(cards_df, y_col='s28')
    
    # log y_train if applicable
    if log_y:
        y = np.log(y)

    # Cross-validate model & predict
    baseline = BaselineModel()
    my_model = GBR_V1() # creates pipeline?
    base_scores = cross_val_score(baseline, X, y, cv=n_folds, verbose=1)
    my_scores = cross_val_score(model, X, y, cv=n_folds, verbose=1)
    print("baseline scores: \n{}".format(base_scores))
    print("my scores: \n{}".format(my_scores))

    # Unlog y_pred if applicable
    # Format & plot results
    
    pass

# TODO NEED TO WRITE PROPER BASELINE MODEL
class BaselineModel(BaseEstimator, RegressorMixin):
    """Baseline Model to evaluate mine against"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Store average prices by rarity for training data
        self.rarity_averages_ = {}

        for rarity in X['rarity'].unique():
            train_mask = X['rarity']==rarity
            rarity_averages_[rarity] = y[train_mask].mean()
        
        return self

    def predict(self, X):
        """ Predict mean prices by rarity, based on training data """
        # Make sure the model has been fit with averages
        try:
            getattr(self, "rarity_averages_")
        except:
            raise AttributeError("rarity_averages_ doesn't exist; make sure you fit a model first")    

        y_preds = pd.Series(X.shape[0])
        
        for rarity in X['rarity'].unique():
            test_mask = X['rarity']==rarity
            y_preds[test_mask] = self.rarity_averages_[rarity]
        
        return y_preds

    def score(self, X, y=None):
        """ Use RMLSE; Root Mean Log Squared Error. Score is -RMLSE, because bigger is better"""
        y_pred = self.predict(X)
        return -rmlse(y_pred, y)

class UnipriceModel(BaseEstimator, TransformerMixin):
     """Model using only recent prices (done already; need to formalize"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Cleave split cards and transforms
        return self

    def transform(self, X):
        df = X.copy()
        return df

class StandardNormalizerModel(BaseEstimator, TransformerMixin):
     """Model using only recent prices (done already; need to formalize"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Cleave split cards and transforms
        return self

    def transform(self, X):
        df = X.copy()
        return df