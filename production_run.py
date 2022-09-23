import data_extractor as data_extractor
import regressor as regressor
import odds as odds


##### Change to argument in code using argparse?
output_fname = 'football_betting_output.csv'
#####

print('\nGetting production data...')
production_df = data_extractor.get_production_data(n_weeks_minus=52, n_weeks_plus=8)
print('Data exracted!')


print('\nSetting config and model features based on current production data...')
features = [f for f in list(production_df) if ('f|team' in f) or ('f|opp' in f)]# or ('f|home' in f)]
production_config = {'fit_intercept':False,
          'target':'p|goals|scored',
          'features':features,
          'weight_col':'f|recency_weight',
          'weight_power':4.5} # This plus n_weeks_minus=52 is best set of hyperparameters
print('Config set!')


print('\nTraining model...')
production_model = regressor.PoissonRegressor(config = production_config)
production_model.train(production_df)
print('Model trained!')
production_model.print_params_pretty()


print('\nGenerating preductions for all games...')
match_preds_dict = production_model.generate_match_preds(production_df, force_home_impact=0.181841)
match_preds_df = regressor.match_preds_to_df(match_preds_dict)
match_preds_df = match_preds_df[['date', 'home_team','away_team','played','p|home_win','p|draw','p|away_win','p|winnings|home_win','p|winnings|draw','p|winnings|away_win','odds|home_win','odds|draw','odds|away_win','true|home_win','true|draw','true|away_win','p|score_1|score','p|score_1|prob','p|score_2|score','p|score_2|prob','p|score_3|score','p|score_3|prob','p|score_4|score','p|score_4|prob','p|score_5|score','p|score_5|prob']]
print('Generated')


print('\nGetting odds for future games and appending')
odds_df = odds.get_odds()
match_preds_df = odds.append_odds_to_df(match_preds_df, odds_df)
match_preds_df.to_csv(output_fname)
print('Appended and saved as:',output_fname)


print('\nUse output to calculate expected winnings ratio (i.e. fraction won on bet on average, >1 is good)\nCalculate like: "ratio = true_probability*(1+odds)"')