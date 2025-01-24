from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, pred)),
            'r2': r2_score(y, pred)
        }