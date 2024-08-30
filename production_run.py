import data_extractor as data_extractor
import regressor as regressor
import odds as odds
import json
import datetime

##### Change to argument in code using argparse?
league_info_fpath = '/Users/dominicbates/Documents/Github/football-score-predictor/league_info/league_info_2024.json'
promoted_prior = True
#####


# Load league info (e.g. names + promoted teams_
with open(league_info_fpath, 'r') as file:
    league_info = json.load(file)

# To store dataframes
league_preds = {}

# Loop through all leagues and get data
for league in list(league_info):

    #try:
    name = league_info[league]['name']
    print('\n\nRunning league: ',name,'('+league+')')
    print('---------------------------------------------')
    
    
    print('- Downloading data')
    production_df = data_extractor.get_production_data(league=league, 
                                                       n_weeks_minus=52, 
                                                       n_weeks_plus=8)

    
    # Priors for promoted or relegated teams
    if league_info[league]['promoted'] is not None:
        print('- Forcing promoted team priors')
        production_df = data_extractor.apply_promoted_prior(production_df, 
                                                            mapping = league_info[league]['promoted'])
    
    if league_info[league]['relegated'] is not None:
        print('- Forcing relegated team priors')
        production_df = data_extractor.apply_promoted_prior(production_df, 
                                                            mapping = league_info[league]['relegated'])
        
    
    # Set params for run 
    print('- Setting config and model features based on current production data...')
    features = [f for f in list(production_df) if ('f|team' in f) or ('f|opp' in f)]# or ('f|home' in f)]
    production_config = {'fit_intercept':False,
              'target':'p|goals|scored',
              'features':features,
              'weight_col':'f|recency_weight',
              'weight_power':4} # This plus n_weeks_minus=52, and forced home/away weight is best set of hyperparameters

    # Model fitting
    print('- Training model...')
    production_model = regressor.PoissonRegressor(config = production_config)
    production_model.train(production_df)
    production_model.print_params_pretty()
    
    # Sample to get predictions
    print('- Generating predictions for all games...')
    match_preds_dict = production_model.generate_match_preds(production_df, force_home_impact=0.181841)
    match_preds_df = regressor.match_preds_to_df(match_preds_dict)
    match_preds_df = match_preds_df[['date', 'home_team','away_team','played','p|home_win','p|draw','p|away_win','p|winnings|home_win','p|winnings|draw','p|winnings|away_win','odds|home_win','odds|draw','odds|away_win','true|home_win','true|draw','true|away_win','p|score_1|score','p|score_1|prob','p|score_2|score','p|score_2|prob','p|score_3|score','p|score_3|prob','p|score_4|score','p|score_4|prob','p|score_5|score','p|score_5|prob']]

    # Get odds from 888sport
    print('- Getting odds for future games and appending')
    odds_df = odds.get_odds(country=league_info[league]['country_888_sport'],
                            league=league_info[league]['league_888_sport'])
    league_preds[name] = odds.append_odds_to_df(match_preds_df, odds_df)
    # Add extrac column for league
    league_preds[name]['league'] = name 
    league_preds[name] = league_preds[name][league_preds[name]['date'].dt.date >= datetime.date.today()]
    match_preds_df.sort_values('date',ascending=False,ignore_index=True).to_csv('football_betting_output_'+name+'.csv')
    print('- Finished')


#     except:
#         Error('Problem with run: '+league_info[league]['name']+', so skipping')
