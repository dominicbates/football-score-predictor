from sklearn import linear_model
import numpy as np


class PoissonRegressor:
    
    def __init__(self, config):
        
        self.min_date = None
        self.max_date = None
        if 'min_date' in list(config):
            self.min_date = config['min_date']
        if 'max_date' in list(config):
            self.max_date = config['max_date']
        if 'features' not in list(config):
            raise ValueError('Need to supply list of features')
        if 'target' not in list(config):
            raise ValueError('Need to supply target')
        self.features = config['features']
        self.target = config['target']    
        
        self.model = linear_model.PoissonRegressor()
        self.model_params = None
    
    def train(self, df):
        df = df[df['f|played'] == 1]
        self.model.fit(df[self.features], df[self.target])
        self.model_params = dict(zip(features,self.model.coef_))
        
    def predict(self, df):
        return self.model.predict(df[features])
