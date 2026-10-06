import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dataset
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

y = np.array([
    35, 40, 45, 52, 58,
    65, 70, 78, 85, 92
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display model information
print("Linear Regression")
print("------------------")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nActual values:")
print(y_test)

print("\nPredicted values:")
print(y_pred)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics")
print("------------------")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Plot regression line
plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual Data")

plt.plot(
    X,
    model.predict(X),
    label="Regression Line"
)

plt.title("Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()