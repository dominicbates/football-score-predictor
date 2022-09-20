import data_extractor as data_extractor
import regressor as regressor

print('\nGetting production data...')
production_df = data_extractor.get_production_data(n_years_minus=2, n_weeks_plus=8)
print(production_df.head())
print('Data exracted!')

print('\nSetting config and model features based on current production data...')
features = [f for f in list(production_df) if ('f|team' in f) or ('f|opp' in f) or ('f|home' in f)]
production_config = {'fit_intercept':False,
          'target':'p|goals|scored',
          'features':features,
          'weight_col':'f|recency_weight'}
print('Config set!')

print('\nTraining model...')
production_model = regressor.PoissonRegressor(config = production_config)
production_model.train(production_df, recency_weight=1.5)
print('Model trained!')

print('\nGenerating preductions for all games...')
match_preds_dict = production_model.generate_match_preds(production_df)
for m in match_preds_dict:
    print(match_preds_dict[m])
    break
match_preds_df = regressor.match_preds_to_df(match_preds_dict)
print(match_preds_df.head())
# match_preds_df.to_csv('test_football_output.csv')
match_preds_df[['date', 'home_team','away_team','played','p|home_win','p|draw','p|away_win']].to_csv('test_football_betting_output.csv')

# NEED TO TURN TO FINAL OUTPUT FOTMAT
print('Generated and saved!')
print('\n\nUse output to calculate ratio (i.e. expected winning fraction on bet, >1 is good)\nCalculate like: "ratio = true_probability*(1+odds)"')