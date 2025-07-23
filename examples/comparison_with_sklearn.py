"""
Comparison with scikit-learn's implementation
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from linear_regression import OrdinaryLeastSquares

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 3)
y = 4 + 3*X[:,0] + 2*X[:,1] - 1*X[:,2] + np.random.randn(100)

# Scikit-learn implementation
sk_model = LinearRegression()
sk_model.fit(X, y)
sk_r2 = sk_model.score(X, y)

# Our implementation
our_model = OrdinaryLeastSquares()
our_model.fit(X, y)
our_r2 = our_model.score(X, y)

print(f"Sklearn R²: {sk_r2:.6f}")
print(f"Our R²:     {our_r2:.6f}")
print(f"Coefficient difference: {np.max(np.abs(sk_model.coef_ - our_model.coef_)):.6f}")