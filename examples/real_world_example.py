"""
Real-world dataset example using Boston housing data
"""
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from linear_regression import GradientDescent


data = load_diabetes()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)


model = GradientDescent(learning_rate=0.01, max_iter=2000, verbose=1)
model.fit(X, y)


print(f"\nTrained on {X.shape[0]} samples with {X.shape[1]} features")
print(f"Final R² score: {model.score(X, y):.4f}")
print(f"First 5 predictions: {model.predict(X[:5])}")
