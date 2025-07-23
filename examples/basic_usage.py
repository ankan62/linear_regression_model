"""
Basic usage example of the linear regression package
"""
import numpy as np
from linear_regression import OrdinaryLeastSquares, GradientDescent

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# OLS Regression
print("Ordinary Least Squares:")
ols = OrdinaryLeastSquares(fit_intercept=True)
ols.fit(X, y)
print(f"Intercept: {ols.intercept_:.4f}, Coefficient: {ols.coef_[0]:.4f}")

# Gradient Descent Regression
print("\nGradient Descent:")
gd = GradientDescent(learning_rate=0.1, max_iter=1000)
gd.fit(X, y)
print(f"Intercept: {gd.intercept_:.4f}, Coefficient: {gd.coef_[0]:.4f}")
print(f"Training iterations: {gd.n_iter_}")