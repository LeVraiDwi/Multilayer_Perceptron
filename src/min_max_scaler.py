import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class min_max_scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
     
    def fit(self, X, y):
        # Pour les ondelettes, il n'y a rien à "apprendre" sur le dataset,
        # donc on se contente de retourner self.
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.y_one_hot = np.column_stack((y == 'B', y == 'M')).astype(int)
        return self
    
    def transform(self, X):
        # On appelle votre fonction existante ici
        ret = (X - self.min) / ((self.max - self.min) + 1e-8)
        print(ret)
        return ret
