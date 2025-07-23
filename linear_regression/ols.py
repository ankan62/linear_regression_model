import numpy as np
from .base import BaseLinearRegression
from .exception import NotFittedError, InvalidInputError
from .utils import validate_input, add_intercept

class OrdinaryLeastSquares(BaseLinearRegression):
    
    def fit(self, X, y) -> 'OrdinaryLeastSquares':
        
        # Validate and convert input
        X, y = validate_input(X, y)
        
        # Add intercept if specified
        if self.fit_intercept:
            X = add_intercept(X)
        
        try:
            weights = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
            )
        
        # Store coefficients
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        
        return self

    def __repr__(self) -> str:
        return (
            f"OrdinaryLeastSquares(fit_intercept={self.fit_intercept})"
        )