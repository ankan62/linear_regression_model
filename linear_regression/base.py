from abc import ABC, abstractclassmethod
import numpy as np 
from .exception import NotFittedError

class BaseLinearRegression(ABC):
    def __init__(self,fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
    @abstractclassmethod
    def fit(self,X,y) -> 'BaseLinearRegression':
        pass

    def predict(self , X) -> np.ndarray:
        if self.coef_ is None:
            raise NotFittedError("Model must be fitted before prediction")
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1,-1)

        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        y = np.array(y)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - (ss_res / (ss_tot + 1e-10))
    
        
