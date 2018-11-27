from model.master_transmuter import *
from model.models import *
from scrape.scraper import *
from query import *

def scraping_progress(rarities):
    connection = connect_mystic()
    
    for rarity in rarities:
        results = connection.execute("select count(*) from {}_price_history_2".format(rarity))
        print('Number of {} price datapoints:'.format(rarity))
        for r in results:
            print(r[0])

        results = connection.execute("select count(distinct cardname) from {}_price_history_2".format(rarity))
        print('Number of {} cards recorded:'.format(rarity))
        for r in results:
            print(r[0])

def test_feature_engineering(cards_df):
    # Test creature feature eng
    # 1 normal creature, one weird creature, one spell
    creature = cards_df[cards_df['name']=='Arcanis the Omnipotent'].iloc[0]
    spell = cards_df[cards_df['name']=='Momentary Blink'].iloc[0]
    weird = cards_df[cards_df['name']=='Tarmogoyf'].iloc[0]
    tests = pd.concat([creature, spell, weird], axis=1).T

    creature_feature = CreatureFeatureTransformer()
    new_feats_df = creature_feature.transform(tests)
    new_feats = ['name']+list(set(new_feats_df.columns) - set(cards_df.columns))
    print(new_feats_df[new_feats])

def query_rarity_dfs(rarity='mythic', version=2, rows=10):
    tablename = rarity+'_price_history_'+str(version)
    connection = connect_mystic()

    query = ("select ph.cardname, ph.setname, ph.timestamp, ph.price "
            "from {0} ph, "
            "     (select ph2.cardname, ph2.setname, max(timestamp) as lastdate "
            "      from {0} ph2 "
            "      group by ph2.cardname, ph2.setname) mr "
            "where ph.timestamp = mr.lastdate "
            "and ph.cardname = mr.cardname "
            "and ph.setname = mr.setname ").format(tablename)

    # Do the thing
    recent_df = pd.read_sql(query, connection)
    connection.close()
    return recent_df.sample(rows)

def chandra_price_check():
    cardname = 'Chandra, Torch of Defiance'
    setname = 'Kaladesh'
    seasons = np.array(pd.read_csv("data/season_dates.csv"))
    tablename = "mythic_price_history_2"
    df = get_twavg_card(cardname, setname, seasons, tablename)
    print(df)

def test_baseline_model():
    cards_df = combine_csv_rarities()
    baseline = BaselineModel()
    baseline.fit(cards_df, cards_df['price'])
    print("Train Baseline Score: {}".format(baseline.score(cards_df, cards_df['price'])))

def test_SpotPriceByRarityGBR():
    cards_df = combine_csv_rarities().sample(100)
    model = SpotPriceByRarityGBR()
    modelname = "SpotPriceByRarityGBR"
    pipe = create_pipeline(model, modelname)
    pipe.fit(cards_df, cards_df['price'])
    print("Train SpotPriceByRarityGBR Score: {}".format(pipe.score(cards_df, cards_df['price'])))

def test_model_comparison():
    cards_df = combine_csv_rarities().sample(100)
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

    run_models_against_baseline([[pipe_a, modelname_a],
                                 [pipe_a1, modelname_a1],
                                 [pipe_b, modelname_b],
                                 [pipe_b1, modelname_b1]], 
                                 cards_df, scorer, n_folds=2)

def test_standard_normalizer():
    """ Predicts Ixalan Prices for season 25. Trains on seasons 1-24 prices for everything """ 

    print("Getting standard format")
    scorer = rmlse_scorer
    std_sets, std_dates = get_standard_format()
    
    print("Getting seasonal prices df")
    X = join_features_seasonal_prices()

    # Setting Ixalan as test set
    X_test = X[X['setname']=="Ixalan"]
    y_test = X_test['s25']

    # Killing stuff after Ixalan
    X = X[X['setname']!="Dominaria"]
    X = X[X['setname']!="Rivals of Ixalan"]
    X = X[X['setname']!="Ixalan"]
    X.drop(columns=['s25','s26','s27','s28'], inplace=True)

    # Performing set exclusion transformation to get final training data
    print("Cleaning X and Y")
    X_train, y_train = csv_cleaner(X, y_col='s24')

    pipe = Pipeline([
        ('BoolToInt', BoolTransformer()),
        ('CreatureFeature', CreatureFeatureTransformer()),
        ('Planeswalker', PlaneswalkerTransformer()),
        ('AbilityCounts', AbilityCountsTransformer()),
        ('Fillna', FillnaTransformer()),
        ('CostIntensity', CostIntensityTransformer()),
        ('DummifyType', TypelineTransformer()),
        ('DummifyColorID', ColorIDTransformer()),
        ('DropFeatures', DropFeaturesTransformer()),
        ('TestFill', TestFillTransformer()),
        ('StandardNormalizerGBR', StandardNormalizerGBR(std_sets_df = std_sets, log_y=True))
    ])
    
    print("Fitting pipeline")
    pipe.fit(X_train, y_train)

    print("Scoring model on Ixalan")
    print(pipe.score(X_test,y_test))

    return pipe

def test_Ixalan_baseline():
    baseline = BaselineModel()
    df = join_features_seasonal_prices()
    X, y = csv_cleaner(df,y_col=['s24','s25'])
    
    seasons = get_seasons(X)
    ix_mask = X['setname']=="Ixalan"
    y_train = y['s24'][~ix_mask]
    y_test = y['s25'][ix_mask]
    X_train = X[~ix_mask].drop(columns=seasons)
    X_test = X[ix_mask].drop(columns=seasons)

    baseline.fit(X_train, y_train)
    print("Baseline score on Ixalan: \n", baseline.score(X_test,y_test))

    return baseline

def plot_Ixalan_model_baseline():
    """ Predicts Ixalan Prices for season 25 using SN-GBR, comparing to baseline""" 
    df = join_features_seasonal_prices()
    X, y = csv_cleaner(df,y_col=['s24','s25'])

    # Drop sets after Ixalan - cheating to count them!
    X = X[X['setname']!="Dominaria"]
    X = X[X['setname']!="Rivals of Ixalan"]
    
    # Setting Ixalan as test set
    ix_mask = X['setname']=="Ixalan"
    y_train = y['s24'][~ix_mask]
    y_test = y['s25'][ix_mask]
    X_train = X[~ix_mask]
    X_test = X[ix_mask]

    print("Getting standard format for pipeline")
    std_sets, std_dates = get_standard_format()

    print("Setting up models")
    pipe = Pipeline([
        ('BoolToInt', BoolTransformer()),
        ('CreatureFeature', CreatureFeatureTransformer()),
        ('Planeswalker', PlaneswalkerTransformer()),
        ('AbilityCounts', AbilityCountsTransformer()),
        ('Fillna', FillnaTransformer()),
        ('CostIntensity', CostIntensityTransformer()),
        ('DummifyType', TypelineTransformer()),
        ('DummifyColorID', ColorIDTransformer()),
        ('DropFeatures', DropFeaturesTransformer()),
        ('TestFill', TestFillTransformer()),
        ('StandardNormalizerGBR', StandardNormalizerGBR(std_sets_df = std_sets, log_y=True))
    ])
    # Drop seasons for baseline 
    seasons = get_seasons(X)
    baseline = BaselineModel()

    print("Fitting Baseline Model")
    baseline.fit(X_train.drop(columns=get_seasons(X_train)), y_train)
    print("Baseline score on Ixalan:")
    print(baseline.score(X_test.drop(columns=get_seasons(X_test)),y_test))

    print("Fitting SN-GBR pipeline")
    pipe.fit(X_train.drop(columns=['s25','s26','s27','s28']), y_train)
    print("SN-GBR scor on Ixalan:")
    print(pipe.score(X_test.drop(columns=['s25','s26','s27','s28']),y_test))
    
    y_pred = baseline.predict(X_test.drop(columns=get_seasons(X_test)))
    baseline_df = format_results(X_test.drop(columns=get_seasons(X_test)), y_pred, y_test)
    y_pred = pipe.predict(X_test.drop(columns=['s25','s26','s27','s28']))
    results_df = format_results(X_test.drop(columns=get_seasons(X_test)), y_pred, y_test)

    plot_residuals_vs_baseline(results_df, baseline_df, title)

    print("SN-GBR feature importances: \n", pipe_feature_imports(pipe))

if __name__ == "__main__":
    # run tessssts
    # test_baseline_model()
    # test_SpotPriceByRarityGBR()
    # test_model_comparison()

    # cards_df = combine_csv_rarities()
    # model_gauntlet(cards_df)
    # test_Ixalan_baseline()
    # test_standard_normalizer()

    plot_Ixalan_model_baseline():
