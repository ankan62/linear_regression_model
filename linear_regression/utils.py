import numpy as np
from typing import Tuple
from .exception import InvalidInputError



def train_test_split(X, y, test_size=0.2, random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    test_samples = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    

def validate_input(X,y) -> Tuple[np.ndarray , np.ndarray]:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1,1)

    #shape validation
    if X.ndim != 2:
        raise InvalidInputError(f"X must be 2D array, got {X.ndim}D ")
    
    if y.ndim != 1 and not (y.ndim == 2 and y.shape[1] == 1):
        raise InvalidInputError(f"y must be 1D array, got shape {y.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise InvalidInputError(
            f"X and y have mismatched sample : X={X.shape[0]},y={y.shape[0]}"
        )
    
    return X, y.squeeze() 

def add_intercept(X: np.ndarray) -> np.ndarray:
    
    return np.column_stack
