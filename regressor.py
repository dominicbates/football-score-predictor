from sklearn import linear_model
import numpy as np


class PoissonRegressor:
    
    def __init__(self, config):
        
        self.min_date = None
        self.max_date = None
        self.fit_intercept = True
        if 'min_date' in list(config):
            self.min_date = config['min_date']
        if 'max_date' in list(config):
            self.max_date = config['max_date']
        if 'features' not in list(config):
            raise ValueError('Need to supply list of features')
        if 'target' not in list(config):
            raise ValueError('Need to supply target')
        if 'fit_intercept' in list(config):
            self.fit_intercept = config['fit_intercept']
        self.features = config['features']
        self.target = config['target']    
        
        self.model = linear_model.PoissonRegressor(alpha=0, fit_intercept=self.fit_intercept)
        self.model_params = None
        self.model_intercept = None
    

    def train(self, df):
        df = df[df['f|played'] == 1]
        self.model.fit(df[self.features], df[self.target])
        self.model_params = dict(zip(self.features,self.model.coef_))
        self.model_intercept = self.model.intercept_


    def predict(self, df):
        return self.model.predict(df[self.features])
    
    
    def print_params_pretty(self):
        if self.model_fit == True:
            
            df = pd.DataFrame()
            df['feature'] = list(model.model_params)
            df['value'] = model.model_params.values()
            attack = df[df['feature'].str.contains('f\|team')].sort_values('value',ascending=False)
            defence = df[df['feature'].str.contains('f\|opp')].sort_values('value',ascending=True)
            other = df[df['feature'].str.contains('f\|home')].sort_values('value',ascending=True)
            print('Attack stats:')
            print(attack.set_index('feature'))
            print('\nDefence stats:')
            print(defence.set_index('feature'))
            print('\nOther features:')
            print(other.set_index('feature'))



    def generate_match_preds(self, df, max_goals = 10, n_draws = 10000):

        # Generate preds
        prediction_means = self.predict(df)
                
        # Blank dict to store stats
        match_preds = {}
            
        # Loop through all matches
        all_matches = list(set(df['match_id']))
        for m_id in all_matches:
            
            m_id_str = str(int(m_id))
            
            # Blank dict
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

            # Win/draw/loss probs
            match_preds[m_id_str]['p|home_win'] = (simulated_goals_h>simulated_goals_a).sum() / n_draws
            match_preds[m_id_str]['p|draw'] = (simulated_goals_h==simulated_goals_a).sum() / n_draws
            match_preds[m_id_str]['p|away_win'] = (simulated_goals_h<simulated_goals_a).sum() / n_draws
            
            # Goal probs
            match_preds[m_id_str]['p|home_goals'] = {}
            match_preds[m_id_str]['p|away_goals'] = {}
            for g in range(max_goals+1):
                match_preds[m_id_str]['p|home_goals'][str(g)] = (simulated_goals_h == g).sum()/n_draws
                match_preds[m_id_str]['p|away_goals'][str(g)] = (simulated_goals_a == g).sum()/n_draws
                
        return match_preds
    
    



