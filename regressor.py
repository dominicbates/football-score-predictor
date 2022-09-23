from sklearn import linear_model
import numpy as np
import pandas as pd


class PoissonRegressor:
    
    def __init__(self, config):
        
        self.min_date = None
        self.max_date = None
        self.fit_intercept = True

        if 'features' not in list(config):
            raise ValueError('Need to supply list of features in config (list of column names)')
        if 'target' not in list(config):
            raise ValueError('Need to supply "target" in config (column name of target variable)')
        if 'weight_col' not in list(config):
            raise ValueError('Need to supply "weight_col" in config (column name of weight feature)')
        if 'weight_power' not in list(config):
            raise ValueError('Need to supply "weight_power" in config (value to raise weight to)')
        if 'fit_intercept' not in list(config):
            raise ValueError('Need to supply "fit_intercept" in config (True or False)')

        self.features = config['features']
        self.target = config['target']
        self.weight_col = config['weight_col']
        self.weight_power = config['weight_power']
        self.fit_intercept = config['fit_intercept']

        self.model = linear_model.PoissonRegressor(alpha=0, fit_intercept=self.fit_intercept)
        self.model_params = None
        self.model_params_split = {}
        self.model_intercept = None
        self.model_fit = False
    
    def train(self, df):
        '''
        Train a model. Recency weight of 0 means all points treated equally, 1 means
        linearly decrease impact with time, 1+ means favour recent games
        '''
        df = df[df['played'] == 1]
        if self.weight_col is None:
            self.model.fit(df[self.features], df[self.target])
        else:
            print('Fitting model with recency weight:',self.weight_power,'(i.e. weight = linear_weight ^ '+str(self.weight_power)+')')
            self.model.fit(df[self.features], df[self.target], sample_weight=df[self.weight_col]**self.weight_power)
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
            print(self.features)

            if np.array([1 for f in self.features if 'f|team' in f]).sum()>0:
                print('Attack stats:')
                attack = df[df['feature'].str.contains('f\|team')].sort_values('value',ascending=False)
                print(attack.set_index('feature'))
            if np.array([1 for f in self.features if 'f|opp' in f]).sum()>0:
                print('\nDefence stats:')
                defence = df[df['feature'].str.contains('f\|opp')].sort_values('value',ascending=True)
                print(defence.set_index('feature'))
            if np.array([1 for f in self.features if 'f|home' in f]).sum()>0:
                print('\nOther features:')
                other = df[df['feature'].str.contains('f\|home')].sort_values('value',ascending=True)
                print(other.set_index('feature'))
            
    def generate_match_preds(self, df, max_goals = 10, n_draws = 10000, n_scores_keep = 10, force_home_impact=None):

        # Generate preds
        prediction_means = self.predict(df)

        # Force shift predictions with pre-fit home impact (0.128041 is fit over 2 year)
        if force_home_impact is not None:
            for n in range(len(prediction_means)):
                prediction_means[n] = np.exp(np.log(prediction_means[n]) + ((df['f|home'][n]-0.5)*force_home_impact))





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
            match_preds[m_id_str]['match_id'] = m_id_str
            match_preds[m_id_str]['date'] = df['date'][m_h].values[0]
            match_preds[m_id_str]['home_team'] = df['team'][m_h].values[0]
            match_preds[m_id_str]['away_team'] = df['team'][m_a].values[0]
            match_preds[m_id_str]['played'] = df['played'][m_h].values[0]
            match_preds[m_id_str]['actual|home_goals'] = df['p|goals|scored'][m_h].values[0]
            match_preds[m_id_str]['actual|away_goals'] = df['p|goals|scored'][m_a].values[0]
            
            # Simulate goals
            # print('SDFSDF\n',prediction_means[m_h], n_draws)
            simulated_goals_h = np.random.poisson(prediction_means[m_h][0], size=n_draws)
            simulated_goals_a = np.random.poisson(prediction_means[m_a][0], size=n_draws)

            # Win probs
            match_preds[m_id_str]['p|home_win'] = (simulated_goals_h>simulated_goals_a).sum() / n_draws
            match_preds[m_id_str]['p|draw'] = (simulated_goals_h==simulated_goals_a).sum() / n_draws
            match_preds[m_id_str]['p|away_win'] = (simulated_goals_h<simulated_goals_a).sum() / n_draws

            # Actual wins etc.
            match_preds[m_id_str]['true|home_win'] = int(match_preds[m_id_str]['actual|home_goals']>match_preds[m_id_str]['actual|away_goals'])
            match_preds[m_id_str]['true|draw'] = int(match_preds[m_id_str]['actual|home_goals']==match_preds[m_id_str]['actual|away_goals']) 
            match_preds[m_id_str]['true|away_win'] = int(match_preds[m_id_str]['actual|home_goals']<match_preds[m_id_str]['actual|away_goals'])
            if match_preds[m_id_str]['played'] != 1:
                match_preds[m_id_str]['true|draw'] = 0 # Fix for unplayed games

                        # Goal probs
            match_preds[m_id_str]['p|home_goals'] = {}
            match_preds[m_id_str]['p|away_goals'] = {}
            for g in range(max_goals+1):
                match_preds[m_id_str]['p|home_goals'][str(g)] = (simulated_goals_h == g).sum()/n_draws
                match_preds[m_id_str]['p|away_goals'][str(g)] = (simulated_goals_a == g).sum()/n_draws
                
            # Most likely scores
            match_preds[m_id_str]['p|scores'] = {}
            for h in match_preds[m_id_str]['p|home_goals']:
                for a in match_preds[m_id_str]['p|away_goals']: # is this right? should be away_goals??
                    match_preds[m_id_str]['p|scores'][str(h)+'-'+str(a)] = (match_preds[m_id_str]['p|home_goals'][h] *
                                                                            match_preds[m_id_str]['p|away_goals'][a])
            # Only keep top N
            scores_df = pd.DataFrame({'scores': list(match_preds[m_id_str]['p|scores']),
                          'probs': list(match_preds[m_id_str]['p|scores'].values())}).sort_values('probs',ascending=False)
            match_preds[m_id_str]['p|scores'] = {}
            for n in range(n_scores_keep):
                match_preds[m_id_str]['p|scores'][scores_df['scores'].iloc[n]] = scores_df['probs'].iloc[n]
            
            
        return match_preds
    
    
    
    
    
def match_preds_to_df(match_preds, n_scores_keep=10):
    
    # Columns to extract
    columns = ['match_id','date','home_team','away_team',
               'played','actual|home_goals','actual|away_goals',
               'p|home_win','p|draw','p|away_win','true|home_win','true|draw','true|away_win']


    new_columns = ['p|score_'+str(n+1)+'|score' for n in range(n_scores_keep)] + \
                  ['p|score_'+str(n+1)+'|prob' for n in range(n_scores_keep)]
    
    # Create blank lists
    final_dict = {}
    for c in columns+new_columns:
        final_dict[c] = []
        
    # Add each game
    for match in match_preds:
        for c in columns:
            final_dict[c].append(match_preds[match][c])

        # Add top score score probabilities
        top_scores = match_preds[match]['p|scores']
        for n in range(n_scores_keep):
            final_dict['p|score_'+str(n+1)+'|score'].append(list(top_scores)[n])
            final_dict['p|score_'+str(n+1)+'|prob'].append(top_scores[list(top_scores)[n]])

    # Turn to dataframe
    final_df = pd.DataFrame(final_dict)

    # Add blank odds columns
    blank_cols = ['odds|home_win','odds|draw','odds|away_win','p|winnings|home_win','p|winnings|draw','p|winnings|away_win']
    for c in blank_cols:
        final_df[c] = 0

    return final_df
    
    




