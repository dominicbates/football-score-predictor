# Football-score-predictor

This is a tool I am building for predicting football scores. 


Data is extracted from the very useful football API https://www.football-data.org/

I am using a very simple Poisson GLM to model football scores (with inputs of team1, team2, home/away)

### 1: data_extractor.py

Contains `download_data()`, which queries the API and gets historical data as json. Also contains `create_df()` and `preprocess_df()` for  processing the raw json and then creating a dataframe ready for trianing. 

`get_production_data()` performs all steps and extracts production data as dataframe


### 2: regressor.py

Contains the `PoissonRegressor` class, which can be used, along with a config file, to train a Poisson GLM, predict expected goals, and also calculate a pdf of goals. Also contains the function `match_preds_to_df()` which turns the dictionary output in to a dataframe

### 3: odds.py

Contains the function `append_odds_to_df()` which gets odds from 888 sport, joins this to a model output dataframe, and also calculates whether odds are under/overvalued

### 4: production_run.py
Runs all steps with production hyper-parameters in order to extract production data, fit a model, and output a final dataframe for use in dashboard/web app

### App: hyperparameter-tuning

Contains some info and plots about hyperparameter tuning in this model. Final production hyperparameters fit the model over fairly short timescales, but we fix the impact of home/away (which was fit over an entire year). This seems to give the best performance after applying the process to many historical dates and assessing how well the model predicts results of the next 2 weeks worth of games

### Possible to do list:


1. Can we make pdf calculation exsct, rather than a random sample?
2. Investigate covariance between two teams scores
3. Create bayesian model + encorporate posteriors in to goals pdf
4. Find and append other data to model (e.g. injury / change in manager info)
5. Better account for promoted team predictions when no data exists
6. Productionise in GCP and output to dashboard (E2 micro?)
