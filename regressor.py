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
    

    def generate_goals_pdf(self, df, max_goals = 10, n_draws = 10000):
    
        df = df.reset_index(level=None, drop=True) # Incase index is not 0,1,2...
        prediction_means = self.predict(df)
        
        # Create blank columns
        for g in range(max_goals+1):
            df['p|prob_'+str(g)+'_goals'] = 0
        
        # Simulate n_goals for each match ('n_draws' times)
        for n in range(len(prediction_means)):
        ####### SUPER INNEFFICIENT AND NOT EXACT - DO IT ANALYTICALLY
            simulated_goals = np.random.poisson(prediction_means[n], size=n_draws)
            
            for g in range(max_goals+1):
                prob_g = (simulated_goals == g).sum()/n_draws # Number of draws with n_goals = g
        #############################################################
                df.loc[n,'p|prob_'+str(g)+'_goals'] = prob_g

         # Return oroginal dataframe with extrac columns
         return df



