from model.master_transmuter import *
from scrape.scraper import *
from query import *
import unit_tests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def fit_basic_pipeline(X_train, X_test, y_train, y_test):

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

def baseline_model(X_train, X_test, y_train, y_test):
    """guesses mean price for everything"""

    y_train_log = np.log(y_train)
    # y_test_log = np.log(y_test)

    avg = y_train_log.mean()

    y_pred_log = np.ones(y_test.shape)*avg
    y_pred = price_corrector(np.exp(y_pred_log))

    results_df = X_test.copy()
    results_df['y_pred'] = y_pred
    results_df['y_test'] = y_test
    results_df['log_diff'] = np.abs(np.log(y_pred+1) - np.log(y_test+1))
    score = log_score(y_pred, y_test)

    return results_df, score

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

    log_ax.scatter(np.log(y_test), np.log(y_pred), label='log preds', alpha=0.5)
    log_ax.scatter(np.log(y_test), np.log(y_test), label='log actuals', alpha=0.5)
    log_ax.set_title('{} log predictions vs actual'.format(title))
    log_ax.set_ylabel('predicted log prices')
    log_ax.set_xlabel('actual log prices')
    log_ax.set_xlim(0,max(np.log(y_test).max(), np.log(y_pred).max()))
    log_ax.set_ylim(0,max(np.log(y_test).max(), np.log(y_pred).max()))

    plt.legend()
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