from model.master_transmuter import *
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
        try:
            my_scores = cross_val_score(model, X, y, cv=n_folds, verbose=2, scoring=scorer, n_jobs=-1)
            score_dict[modelname] = my_scores
        except:
            print("{} errored during cross_val".format(modelname))
            score_dict[modelname] = ['Errored during cross_val']
    print("baseline scores: \n\t{}".format(base_scores))
    print("baseline average: \n\t{}".format(np.mean(base_scores)))

    for modelname, scores in score_dict.items():
        print("{MODEL} scores: \n\t{SCORES}".format(MODEL=modelname, SCORES=scores))
        print("{MODEL} average: \n\t{SCORES}".format(MODEL=modelname, SCORES=np.mean(scores)))

    # Format & plot results
    return score_dict

def model_gauntlet(cards_df):
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
    pipe_b1 = create_pipeline(model_b1, modelname_b1)

    run_models_against_baseline([#[pipe_a, modelname_a],
                                 #[pipe_a1, modelname_a1],
                                 [pipe_b, modelname_b],
                                 [pipe_b1, modelname_b1]], 
                                 cards_df, scorer, n_folds=5)

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
            elif rarity == 'mythic':
                y_pred[test_mask] = self.rarity_averages_['rare']
            else:
                y_pred[test_mask] = self.rarity_averages_['uncommon']
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
                 log_y=False, rarities=['mythic', 'rare', 'uncommon', 'common'],
                 rarity_baseline={'mythic':10,'rare':1.5,'uncommon':0.5,'common':0.2}):
        self.base_weight = base_weight
        self.model = model
        self.log_y = log_y
        self.rarities = rarities
        self.rarity_baseline = rarity_baseline
        self.rarity_models_ = {}

    def fit(self, X_train, y_train):
        X = X_train.copy()
        y = y_train.copy()

        if self.log_y:
            for key, value in self.rarity_baseline.items():
                self.rarity_baseline[key] = np.log(value)
            y = np.log(y)
        
        self.train_rarities_ = [x for x in X.columns if x.startswith('rarity_')]

        for rarity in self.train_rarities_:
            train_mask = X[rarity]==1
            rarity_model = clone(self.model)
            rarity_model.fit(X[train_mask], y[train_mask])
            self.rarity_models_[rarity] = rarity_model
        
        return self

    def predict(self, X):
        """ Pick fitted GBR model by rarity """
        try:
            getattr(self, "rarity_models_")
        except AttributeError:
            raise RuntimeError("rarity_models_ doesn't exist; make sure you fit a model first")    

        y_pred = np.ones(X.shape[0])

        test_rarities = [x for x in X.columns if x.startswith('rarity_')]

        for rarity in test_rarities:
            test_mask = X[rarity]==1
            if test_mask.sum():
                # If we have a rarity model, fit it; otherwise, use baseline assumption from init 
                try:
                    y_pred[test_mask] = self.rarity_models_[rarity].predict(X[test_mask])
                except:        
                    y_pred[test_mask] = self.rarity_baseline[rarity]

        if self.log_y:
            y_pred = np.exp(y_pred)

        # Set floor to GBR predictions
        y_pred[y_pred<0.1]=0.1
        return y_pred

    def score(self, X, y):
        """ Use RMLSE; Root Mean Log Squared Error. Score is -RMLSE, because bigger is better"""
        y_pred = self.predict(X)
        return -rmlse(y_pred, y)

class StandardNormalizerGBR(BaseEstimator, RegressorMixin):
    """ Uses price history of rarities across standard season to normalize power """
    def __init__(self, model=GradientBoostingRegressor(), base_weight=0,
                 log_y=False, rarities=['mythic', 'rare', 'uncommon', 'common'],
                 std_sets_df=None):
        self.base_weight = base_weight
        self.model = model
        self.log_y = log_y
        self.rarities = rarities
        self.std_sets_df = std_sets_df
        self.standard_price_xfmr = StandardPriceTransformer(self.std_sets_df)

    def _drop_seasons(self, df):
        """ Returns df with season features dropped. Used after getting season attrs, before fitting X """
        seasons = list(self.std_sets_df.columns)
        return df.drop(columns=seasons)

    def _predict_next_standard_market(self, std_prices_df, next_sets):
        """ Fits linear regression to standard market trend to predict size at next season, given standard legal set count """
        # x variables: season num, sin(num sets), interaction 
        seasons = list(self.std_sets_df.columns)

        X_train = pd.DataFrame()
        X_train['set_count'] = self.std_sets_df.sum().reset_index(drop=True)
        X_train['season'] = X_train.index + 1

        y_train = std_prices_df[seasons].sum()

        std_xfmr = StandardSeasonTransformer()
        X_train_prime = std_xfmr.transform(X_train)

        lr = LinearRegression()
        lr.fit(X_train_prime, y_train)

        X_test = pd.DataFrame({'set_count':[next_sets], 'season':[len(seasons)+1]})
        X_test_prime = std_xfmr.transform(X_test)

        y_pred = lr.predict(X_test_prime)
        return y_pred

    def _season_attrs(self):
        """ Calculates, sets attributes to scale y during fitting process """
        spt = StandardPriceTransformer(self.std_sets_df)
        ptpt = PriceToPowerTransformer()
        ptpt.fit(y)
        self.price_transformer_ = ptpt
        ptpt.transform(y)
        pass

    def fit(self, X_train, y_train):
        X = X_train.copy()
        y_price = y_train.copy()

        # get standard prices
        std_prices_df = self.standard_price_xfmr.transform(X)
        
        # Transform prices to power
        self.ptpt_ = PriceToPowerTransformer()
        y_power = self.ptpt_.fit_transform(std_prices_df, y_price)

        # log if applicable
        if self.log_y:
            y_power = np.log(y_power)
        
        # drop seasonal price features and fit GBR
        X = self._drop_seasons(X)
        self.model.fit(X, y_power)

        return self

    def predict(self, X):
        """ Pick fitted GBR model by rarity """
        try:
            getattr(self, "rarity_models_")
        except AttributeError:
            raise RuntimeError("rarity_models_ doesn't exist; make sure you fit a model first")    

        y_pred = np.ones(X.shape[0])



        if self.log_y:
            y_pred = np.exp(y_pred)

        # Set floor to GBR predictions
        y_pred[y_pred<0.1]=0.1
        return y_pred

    def score(self, X, y):
        """ Use RMLSE; Root Mean Log Squared Error. Score is -RMLSE, because bigger is better"""
        y_pred = self.predict(X)
        return -rmlse(y_pred, y)
