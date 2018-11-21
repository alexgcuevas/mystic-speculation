from model.master_transmuter import *
from scrape.scraper import *
from query import *
import unit_tests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
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
    score = log_score(y_pred, y_test)

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

def log_score(y_pred,y_test):
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

def run_model_against_baseline(model, raw_df, log_y=True, n_folds=5):
    # Run data through cleaner
    
    # log y_train if applicable
    
    # Cross-validate model & predict
    
    # Unlog y_pred if applicable

    # Format & plot results
    
    pass