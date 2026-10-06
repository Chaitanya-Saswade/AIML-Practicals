import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
np.random.seed(42)

X = np.linspace(0, 10, 100).reshape(-1, 1)

y = (
    2 * X.flatten() ** 2
    + 3 * X.flatten()
    + np.random.normal(0, 20, 100)
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Polynomial degree
degree = 10

# Polynomial Linear Regression
linear_model = make_pipeline(
    PolynomialFeatures(degree),
    LinearRegression()
)

# Ridge Regression
ridge_model = make_pipeline(
    PolynomialFeatures(degree),
    Ridge(alpha=10)
)

# Lasso Regression
lasso_model = make_pipeline(
    PolynomialFeatures(degree),
    Lasso(alpha=1, max_iter=10000)
)

# Train models
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Predictions
linear_pred = linear_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)


# Evaluation function
def evaluate_model(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(name)
    print("MSE:", round(mse, 2))
    print("R2 Score:", round(r2, 2))
    print()


print("Regularization Comparison")
print("=========================")

evaluate_model(
    "Polynomial Linear Regression",
    y_test,
    linear_pred
)

evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_pred
)

evaluate_model(
    "Lasso Regression",
    y_test,
    lasso_pred
)


# Plot predictions
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X_plot,
    linear_model.predict(X_plot),
    label="Polynomial Regression"
)

plt.plot(
    X_plot,
    ridge_model.predict(X_plot),
    label="Ridge Regression"
)

plt.plot(
    X_plot,
    lasso_model.predict(X_plot),
    label="Lasso Regression"
)

plt.title("Regularization: Ridge and Lasso")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()