# football-score-predictor

<img src="./images/logo.png" alt="Alt text" width="200" style="display: block; margin: 0 auto;">

This repository contains a code to model win/draw/loss probabilities across most major football leagues (premier league & championship in England, along with the 1st league in germany, france, brazil, spain, netherlands, and portugal). The model applies [Poisson linear regression](https://en.wikipedia.org/wiki/Poisson_regression) to historical results, accounting for importance of recent vs long term form in order to predict the outcome of future games.

There is code to automatically download future fixtures, predict win/draw/loss probabilities, and also join to 888sport odds, to work out if particular odds are over/under valued. Has been succesfully applied to make money from betting sites, however would advise applying at your own risk, given the uncertainty and biases around this kind of modelling. Documentation is currently mostly for personal use, however anyone else is welcome to clone the repo and run as desired.

## 1. The Model

The hypothesis underlying this model is that goal scoring is an inherrently Poisson process, e.g. similar to number of raindrops falling on a particular area of ground, or number of cars passing on a quiet road. Specifically, we assume that the number of team goals scored in a football game is drawn from a Poisson distribution, with a mean dictated by the ability of the two teams (and possibly some other features). This assumption has been subsequently tested here, where we find that the distribution of goals for a particular predicted mean ***perfectly matches a Poission distribution*** (at least within the limits of our dataset size).

Compared to some previous approaches, we *don't* manually create features for recent form, but rather we create features representing each attacking and defending team, and model form as part of the fitting process. In Poisson linear regression, the expected number of goals, $G$, in a game can be modelled as:

$$log(G) = {\bf X \beta} + \beta_{0}$$

where $X$ represents our model features, and $\beta$ the weight assigned to each. Our $X$ is an array representing both the attacking team (e.g. `[0,1,0,...]`) and defensive team (e.g. `+ [1,0,0,...]`) for a particular game. We also add a single feature to model the impact of home/away. Further features could be added representing injuries, manager changes etc., however extracting clean historical data for these is very difficult, so we limit this to just teams and home/away. 

Given this model, we can fit to historical results, such that the fitted $\beta_{i}$ now contains the attacking strength and defensive strength of each team in the dataset. To predict the likely results of a future game, we can then set the correct binary features for attacking and defensive teams and read the distribuition mean (doing this twice to get each teams goals). Given these two different Poisson distributions, we can simulate many games to work out the fraction that are won/drawn/lost.

To make sure we are not averaging over outdated results, which may not now be representative (e.g. if a team has improved recently, signed new players, has injuries etc.), we specifically model the importance of recent vs historical results as part of the fitting process. To do this we define a range, $r$, to fit over e.g. (only fit over the last 10 games), and also define a weight, $w$ which is applied to a linearly increasing array the same length as the number of games included (e.g. `[0.1,0.2,...,0.9,1.0] ^ w`, where 0.1 corresponds to the 10th most recent game). This array is raised to the power, $w$, such that for a vanishingly _small_ weight ($w=0$), all games are treated equally, and for a very  _large_ one ($w>>1$), only the most recent game is weighted in the fit. 

We fit models over all many different values of $r$ and $w$ for ~5 years worth of premier league data to see which produces the most accurate game predictions (some plots of this process are shown [here](https://github.com/dominicbates/football-score-predictor/tree/master/hyperparameter-tuning)). We find that setting $r$ to the most recent 1 year's games, and $w=4$ (heavily weighting recent games) produces the most accuract model, and hence best represents current form. 


## 2. The Code

Code is split in to a few different scripts, representing the different distinct steps in downloading data, modelling, and calculating odds. A production run script has been also added, which does all steps needed for running the code in production. Some file paths may need to be edited if running on your own machine, as code has not been fully productionised, however this should be limited to edting a couple of lines in `production_run.py`.

### data_extractor.py

Contains `download_data()`, which queries the API and gets historical data as json. Also contains `create_df()` and `preprocess_df()` for  processing the raw json and then creating a dataframe ready for trianing. 

`get_production_data()` performs all steps and extracts production data as dataframe. Optionally a prior can be added on prompted teams using `apply_prompted_prior()` to this output, editing the dataframe to force a prior which assumes similar performance to last years promoted teams (e.g. assume Leicster perform similarly to Burnley and gradually correct this as new data appears).


### regressor.py

Contains the `PoissonRegressor` class, which can be used, along with a config file, to train a Poisson GLM, predict expected goals, and also calculate a pdf of goals. Also contains the function `match_preds_to_df()` which turns the dictionary output in to a dataframe

### odds.py

Contains the function `append_odds_to_df()` which gets odds from 888 sport for a particular league, joins this to a model output dataframe, and also calculates whether odds are under/overvalued based on the computed win/draw/loss probabilities. Matching works fine for premier league and championship, however currently some problems with some team names not matching across other leagues (needs some manual adjustment - likely to do with non-latin characters).

### production_run.py
Runs all steps with production hyper-parameters in order to extract production data, fit a model, and output a final dataframe with model predictions for upcoming games along with undervalued odds. Currently creates a seperate csv for each league.

### hyperparameter-tuning/

Contains some info and plots about hyperparameter tuning in this model. Final production hyperparameters fit the model over fairly short timescales, but we fix the impact of home/away (which was fit over an entire year). This seems to give the best performance after applying the process to many historical dates and assessing how well the model predicts results of the next 2 weeks worth of games

## 3. To do list:

1. Fix team name matching across non-english leagues.
2. Productionise in streamlit?
3. Measure covariance between the goals of both teams to see if distributions are uncorrelated. If there is a significant correlation, we can then account for this in our monte carlo sampling process
4. Get this working for several leagues  and streamline code to computer for all on one step (seperate models).
5. Could build a seperate model or process to calculate: For player X in team Y, what fraction of Ys goals go through them (then can multiply this by team stats to get player goal probabilities)
6. Can we make pdf calculation exact, rather than a random sample. 
7. Create bayesian model + encorporate posteriors in to goals pdf
8. Find and append other data to model (e.g. injury / change in manager info)
9. Better account for promoted team predictions when no data exists (partially done)


