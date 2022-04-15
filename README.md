# Football-score-predictor

This is a tool I am building for predicting football scores. 

Data is extracted from the very useful football API https://www.football-data.org/

I am using a very simple Poisson GLM to model football scores (with inputs of team1, team2, home/away)

### 1. data_extractor.py

Contains `download_data()`, which queries the API and gets historical data as json. Also contains `create_df()` and `preprocess_df()` for  processing the raw json and then creating a dataframe ready for trianing


### 2. regressor.py

Contains the `PoissonRegressor` class, which can be used, along with a config file, to train a Poisson GLM, predict expected goals, and also calculate a pdf of goals


### To Do:

1. Create score predictor function (randomly sample two pdfs and compute socre probabilities + win/draw/loss probability) - think about where this should go
2. Add negative option to regression class (rename GoalsRegressor) and add to random draw / score predictor function (self.method = 'poisson' or 'negative_binomial') 
3. Can we make pdf calculation exsct, rather than a random sample
4. Create model validation approach (training/test sets)
5. Improve model performance (features, time window, test/train validation)
6. Investigate covariance between two teams scores
7. Create bayesian model + encorporate posteriors in to goals pdf
8. Productionise: (Create file to extract data + train prod models from command line to file, automate push to google sheet, build dashboard)
