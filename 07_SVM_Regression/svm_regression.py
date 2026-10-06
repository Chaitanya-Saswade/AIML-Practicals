import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create dataset
np.random.seed(42)

X = np.arange(1, 101).reshape(-1, 1)

y = (
    3 * X.flatten()
    + 10 * np.sin(X.flatten() / 5)
    + np.random.normal(0, 5, 100)
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Standardize input data
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Create SVM Regression model
model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

# Train model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("SVM Regression")
print("--------------")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Sort test values for a smooth graph
order = np.argsort(X_test.flatten())

# Plot results
plt.figure(figsize=(10, 6))

plt.scatter(
    X_test,
    y_test,
    label="Actual"
)

plt.plot(
    X_test.flatten()[order],
    y_pred[order],
    label="SVR Prediction"
)

plt.title("Support Vector Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)

plt.show()