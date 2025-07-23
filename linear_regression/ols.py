import numpy as np
from .base import BaseLinearRegression
from .exception import NotFittedError, InvalidInputError
from .utils import validate_input, add_intercept

class OrdinaryLeastSquares(BaseLinearRegression):
    """Linear Regression using Ordinary Least Squares"""
    
    def fit(self, X, y) -> 'OrdinaryLeastSquares':
    
    
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
    
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Validate input shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different number of samples: {X.shape[0]} vs {y.shape[0]}")
        
        # Add intercept if specified
        if self.fit_intercept:
            X = add_intercept(X)
        
        try:
            # Calculate closed-form solution
            weights = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Matrix is singular. Try reducing features or adding regularization"
            ) from e
        
        # Store coefficients
        if self.fit_intercept:
            self.intercept_ = weights[0, 0] if weights.ndim > 1 else weights[0]
            self.coef_ = weights[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.flatten()
        
        self._is_fitted = True
        return self

    def __repr__(self) -> str:
        return f"OrdinaryLeastSquares(fit_intercept={self.fit_intercept})"
