# Linear Regression from Scratch

A clean and modular implementation of linear regression algorithms that follows the Scikit-learn API design. Ideal for learning purposes, experimentation, and simple regression tasks.

## Features

- **Two Algorithm Options**:
  - **Ordinary Least Squares (Closed-form solution)**: Computes the best-fit parameters analytically using matrix operations.
  - **Gradient Descent (Iterative optimization)**: Minimizes the loss function through iterative parameter updates, suitable for large datasets.

- **Scikit-learn Compatible API**:
  - Methods follow the `fit()`, `predict()`, and `score()` pattern, making it easy to switch between this implementation and Scikit-learn’s.
  - Accepts input and output data in the same format as Scikit-learn models (NumPy arrays).

- **Additional Utilities**:
  - Input validation to ensure proper data shape and type.
  - Training progress visualization for gradient descent via the `verbose` flag.
  - Implements `score()` method to return the coefficient of determination (R² score), allowing model evaluation in a standard way.

## Installation

You can directly install the package using pip:

```bash
pip install git+https://github.com/ankan62/linear_regression_model.git
