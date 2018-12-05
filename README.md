# Mystic Speculation
_Predicting Magic: The Gathering card prices with card feature data and seasonal price trends_


![title](http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=126156&type=card)

## Overview

This project aims to predict prices of Magic: The Gathering cards using card feature data and price history. Card feature data is provided by Scryfall's API, and price history was scraped from the web. (scrape at your own risk - it's often better to ask for such data!) 

## Structure

* `scrape/` contains code to access card feature data from Scryfall's API and for scraping price history from the web
* `data/` where local card data is downloaded to, generally in csv format
* `model/` contains pipelines, transformers, models, and plotting methods
* `figures/` is a place to store output plots from visualizations and model results
* `root`
  * `query.py` contains code to query from the price history database to set up local csvs, etc. for manipulation in pandas and model training.
  * `test_vs_notebook.py` is a scratchpad where I prototyped and tested methods, queries, models etc. in visual studio 
  * `unit_tests.py` contains structured tests of methods, as well as code to run and visualize models of particular data. 


## How-to

1. Start with getting your data; scryfall is great for the card features, and you can get price history from wherever you deem appropriate, as long as unique (cardname,setname) combinations are clear for each card, or you have a specific card id you can peg to from scryfall. 
2. Run appropriate unit tests to make sure that code works with the database you set up, and write / modify your own methods as needed.
3. Experiment; try out different models and pipelines; feature engineer; and most importantly, enjoy! 

## Findings

My best predictive model (RMLSE = 0.3 tested on Ixalan's cards, using cards printed before that) were these determinants of price:
* Tournament seasonality
* Rarity
* Engineered features
  * Converted mana cost : average p/t ratio
    * eg Ernam would be cmc(3F) / avg(4,5) = 4/4.5 = 0.89
  * number of abilities
    * activated
    * triggered
    * total ability blocks

The tournament seasonality represented how prices of cards vary with the Standard format, the hypothesis being that the mtg "market" has a "segment" which can offer a certain support in terms of total cost of legal cards, and I normalized my predicted set prices based on a prediction of what total set price the Standard format could bear at time t + 1. 

The rarity represents economic scarcity of cards, which is built into the supply; I started with baseline figures for these and updated them with training data, increasing weights for data closer to the present, and finally scaled price predictions according to this, transforming them instead into a "power" metric. This seasonal trend & rarity normalized power metric is what my model actually predicts, and predictions are transformed into prices afterwards. 

### Future Work

There is much more that could be done to improve this project; both to improve the model, and to improve accessibility of the code for interested users to work off of. 

For model improvement, the main things I would look at next are an n-gram featurization of the card text (to find more specific abilities that improve card power), as well as models to account for eternal format demand drivers. As a stretch goal, accounting for casual and collector value of cards could improve the model even further.

For accessibiliy, I am working on adding a web interface, as well as the ability to easily update and retrain the model as new sets come out.
