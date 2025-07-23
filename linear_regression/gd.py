import numpy as np
from .base import BaseLinearRegression
from .exception import NotFittedError

class GradientDescent(BaseLinearRegression):
    """Linear Regression using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, 
                 fit_intercept=True, random_state=None, verbose=0):
        super().__init__(fit_intercept=fit_intercept)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history_ = []
        self.n_iter_ = 0
    
    def fit(self, X, y) -> 'GradientDescent':
       
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
    
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
      
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.coef_ = np.random.randn(X.shape[1], 1)
        
        
        for i in range(self.max_iter):
            
            y_pred = X @ self.coef_
            error = y_pred - y
            
            
            gradient = (X.T @ error) / len(X)
            
          
            self.coef_ -= self.learning_rate * gradient
            
         
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
            self.intercept_ = self.coef_[0, 0]
            self.coef_ = self.coef_[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = self.coef_.flatten()
        
        return self
