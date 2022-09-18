from sklearn import linear_model
import numpy as np
import pandas as pd


class PoissonRegressor:
    
    def __init__(self, config):
        
        self.min_date = None
        self.max_date = None
        self.fit_intercept = True
        #if 'min_date' in list(config):
        #    self.min_date = config['min_date']
        #if 'max_date' in list(config):
        #    self.max_date = config['max_date']
        if 'features' not in list(config):
            raise ValueError('Need to supply list of features in config "features"')
        if 'target' not in list(config):
            raise ValueError('Need to supply target in config "target"')
        if 'fit_intercept' not in list(config):
            raise ValueError('Need to supply "fit_intercept" in config (True or False)')
            
        self.features = config['features']
        self.target = config['target']    
        self.fit_intercept = config['fit_intercept']

        self.model = linear_model.PoissonRegressor(alpha=0, fit_intercept=self.fit_intercept)
        self.model_params = None
        self.model_params_split = {}
        self.model_intercept = None
        self.model_fit = False
    
    def train(self, df):
        df = df[df['f|played'] == 1]
        self.model.fit(df[self.features], df[self.target])
        self.model_params = dict(zip(self.features,self.model.coef_))
        self.model_intercept = self.model.intercept_
        self.model_fit = True

    def predict(self, df):
        return self.model.predict(df[self.features])
    
    def print_params_pretty(self):
        if self.model_fit == True:
            
            df = pd.DataFrame()
            df['feature'] = list(self.model_params)
            df['value'] = self.model_params.values()
            attack = df[df['feature'].str.contains('f\|team')].sort_values('value',ascending=False)
            defence = df[df['feature'].str.contains('f\|opp')].sort_values('value',ascending=True)
            other = df[df['feature'].str.contains('f\|home')].sort_values('value',ascending=True)
            print('Attack stats:')
            print(attack.set_index('feature'))
            print('\nDefence stats:')
            print(defence.set_index('feature'))
            print('\nOther features:')
            print(other.set_index('feature'))
            
    def generate_match_preds(self, df, max_goals = 10, n_draws = 10000, n_scores_keep = 10):

        # Generate preds
        prediction_means = self.predict(df)
                
        # Blank dict to store stats
        match_preds = {}
            
        # Loop through all matches
        all_matches = list(set(df['match_id']))
        for m_id in all_matches:
            
            # Blank dict
            m_id_str = str(int(m_id))
            match_preds[m_id_str] = {}
            
            # Rows
            m_h = (df['match_id'] == m_id) & (df['f|home'] == 1)
            m_a = (df['match_id'] == m_id) & (df['f|home'] == 0)
            
            # Match stats
            match_preds[m_id_str]['date'] = str(df['date'][m_h].values[0])
            match_preds[m_id_str]['home_team'] = df['team'][m_h].values[0]
            match_preds[m_id_str]['away_team'] = df['team'][m_a].values[0]
            match_preds[m_id_str]['played'] = df['played'][m_h].values[0]
            match_preds[m_id_str]['actual|home_goals'] = df['p|goals|scored'][m_h].values[0]
            match_preds[m_id_str]['actual|away_goals'] = df['p|goals|scored'][m_a].values[0]
            
            # Simulate goals
            simulated_goals_h = np.random.poisson(prediction_means[m_h], size=n_draws)
            simulated_goals_a = np.random.poisson(prediction_means[m_a], size=n_draws)

            # Win probs
            match_preds[m_id_str]['p|home_win'] = (simulated_goals_h>simulated_goals_a).sum() / n_draws
            match_preds[m_id_str]['p|draw'] = (simulated_goals_h==simulated_goals_a).sum() / n_draws
            match_preds[m_id_str]['p|away_win'] = (simulated_goals_h<simulated_goals_a).sum() / n_draws
            
            # Goal probs
            match_preds[m_id_str]['p|home_goals'] = {}
            match_preds[m_id_str]['p|away_goals'] = {}
            for g in range(max_goals+1):
                match_preds[m_id_str]['p|home_goals'][str(g)] = (simulated_goals_h == g).sum()/n_draws
                match_preds[m_id_str]['p|away_goals'][str(g)] = (simulated_goals_a == g).sum()/n_draws
                
            # Most likely scores
            match_preds[m_id_str]['p|scores'] = {}
            for h in match_preds[m_id_str]['p|home_goals']:
                for a in match_preds[m_id_str]['p|home_goals']:
                    match_preds[m_id_str]['p|scores'][str(h)+'-'+str(a)] = (match_preds[m_id_str]['p|home_goals'][h] *
                                                                            match_preds[m_id_str]['p|away_goals'][a])
            # Only keep top N
            scores_df = pd.DataFrame({'scores': list(match_preds[m_id_str]['p|scores']),
                          'probs': list(match_preds[m_id_str]['p|scores'].values())}).sort_values('probs',ascending=False)
            match_preds[m_id_str]['p|scores'] = {}
            for n in range(n_scores_keep):
                match_preds[m_id_str]['p|scores'][scores_df['scores'].iloc[n]] = scores_df['probs'].iloc[n]
            
            
        return match_preds
    
    
   

    



