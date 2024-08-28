# Football-score-predictor

This is a tool I have built for predicting football scores and win/draw/loss probabilities. 

Data is extracted from the very useful football API https://www.football-data.org/ I am currently only extracting premier league games, but other leagues could easily be fit only a few small changes to the extraction query, and odds scipts.

I am using a very simple Poisson GLM to model the number of goals scored by each team (with inputs of team1, team2, home/away). I also have hyperparameters describing how to weight recent vs historical games in the fit. I am computing the probability of a win/loss/draw for each game, and most likely results by sampling from these two Poisson distributions.

Model has been used to successfully make money from betting sites, however I would advise applying at your own risk, given the uncertainty and biases around this kind of modelling. 

### 1: data_extractor.py

Contains `download_data()`, which queries the API and gets historical data as json. Also contains `create_df()` and `preprocess_df()` for  processing the raw json and then creating a dataframe ready for trianing. 

`get_production_data()` performs all steps and extracts production data as dataframe. Optionally a prior can be added on prompted teams using `apply_prompted_prior()` to this output, editing the dataframe to force a prior which assumes similar performance to last years promoted teams.


### 2: regressor.py

Contains the `PoissonRegressor` class, which can be used, along with a config file, to train a Poisson GLM, predict expected goals, and also calculate a pdf of goals. Also contains the function `match_preds_to_df()` which turns the dictionary output in to a dataframe

### 3: odds.py

Contains the function `append_odds_to_df()` which gets odds from 888 sport, joins this to a model output dataframe, and also calculates whether odds are under/overvalued

### 4: production_run.py
Runs all steps with production hyper-parameters in order to extract production data, fit a model, and output a final dataframe for use in dashboard/web app

### `hyperparameter-tuning/`

Contains some info and plots about hyperparameter tuning in this model. Final production hyperparameters fit the model over fairly short timescales, but we fix the impact of home/away (which was fit over an entire year). This seems to give the best performance after applying the process to many historical dates and assessing how well the model predicts results of the next 2 weeks worth of games

### Possible to do list:

1. Productionise in streamlit?
2. Measure covariance between the goals of both teams to see if distributions are uncorrelated. If there is a significant correlation, we can then account for this in our monte carlo sampling process
3. Get this working for several leagues  and streamline code to computer for all on one step (seperate models).
4. Could build a seperate model or process to calculate: For player X in team Y, what fraction of Ys goals go through them (then can multiply this by team stats to get player goal probabilities)
5. Can we make pdf calculation exact, rather than a random sample. 
6. Create bayesian model + encorporate posteriors in to goals pdf
7. Find and append other data to model (e.g. injury / change in manager info)
8. Better account for promoted team predictions when no data exists (partially done)


