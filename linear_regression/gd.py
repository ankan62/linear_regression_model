import numpy as np
from typing import Union, Optional
from .base import BaseLinearRegression
from .exception import NotFittedError
from .utils import validate_input, add_intercept

class GradientDescent(BaseLinearRegression):
    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        super().__init__(fit_intercept=fit_intercept)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.n_iter_ = 0
        self.loss_history_ = []

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'GradientDescent':
       
        X, y = validate_input(X, y)
        y = y.reshape(-1, 1)
        
        if self.fit_intercept:
            X = add_intercept(X)

        # Initialize weights
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.coef_ = np.random.randn(X.shape[1], 1)
        
        # Gradient Descent
        for i in range(self.max_iter):
            # Calculate predictions and error
            y_pred = X @ self.coef_
            error = y_pred - y
            
            # Calculate gradient (using MSE loss)
            gradient = (X.T @ error) / len(X)
            
            # Update weights
            self.coef_ -= self.learning_rate * gradient
            
            # Calculate and store loss
            loss = (error ** 2).mean()
            self.loss_history_.append(loss)
            self.n_iter_ = i + 1
            
            
            if self.verbose > 0 and i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
            
            
            if i > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                if self.verbose > 0:
                    print(f"Converged at iteration {i}")
                break

        if self.fit_intercept:
            self.intercept_ = self.coef_[0][0]
            self.coef_ = self.coef_[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.coef_.flatten()
            
        return self

    def __repr__(self) -> str:
        return (f"GradientDescent(fit_intercept={self.fit_intercept}, "
                f"learning_rate={self.learning_rate}, max_iter={self.max_iter})")
    
    def score(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:

        y_pred = self.predict(X)
        y_true = np.array(y)
        
        # Calculate total sum of squares
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        
        # Calculate residual sum of squares
        ss_res = ((y_true - y_pred) ** 2).sum()
        
        # Handle case when ss_tot is 0 (perfect fit)
        if ss_tot == 0:
            return 1.0
        
        return 1 - (ss_res / ss_tot)