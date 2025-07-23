import numpy as np
from .base import BaseLinearRegression
from .exception import NotFittedError

class OrdinaryLeastSquares(BaseLinearRegression):
    """Linear Regression using Ordinary Least Squares"""
    
    def fit(self, X, y) -> 'OrdinaryLeastSquares':
        
       
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)  
        
       
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        try:
           
            XT = X.T  # Explicit transpose operation
            self.coef_ = np.linalg.inv(XT @ X) @ XT @ y
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Matrix is singular. Try reducing features or adding regularization"
            ) from e
        
       
        if self.fit_intercept:
            self.intercept_ = self.coef_[0][0]
            self.coef_ = self.coef_[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.coef_.flatten()
        
        return self

    def predict(self, X) -> np.ndarray:
        """Make predictions"""
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ np.vstack([self.intercept_, self.coef_])
