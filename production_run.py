import data_extractor as data_extractor
import regressor as regressor

print('\nGetting production data...')
production_df = data_extractor.get_production_data(n_years_minus=2, n_weeks_plus=8)
print('Data exracted!')

print('\nSetting config and model features based on current production data...')
features = [f for f in list(production_df) if ('f|team' in f) or ('f|opp' in f) or ('f|home' in f)]
production_config = {'fit_intercept':False,
          'target':'p|goals|scored',
          'features':features}
print('Config set!')

print('\nTraining model...')
production_model = regressor.Regressor(config = production_config)
production_model.train(production_df)
print('Model trained!')

print('\nGenerating preductions for all games...')
match_preds_dict = production_model.generate_match_preds()

# NEED TO TURN TO FINAL OUTPUT FOTMAT
print('Generating preductions for all games...')
