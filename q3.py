import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# 1. Load Data
train_data = pd.read_csv("../data/training_data.csv")
test_data = pd.read_csv("../data/test_data.csv")
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 2.  Best Model Model 3
best_params = {
    'criterion': 'squared_error',
    'splitter': 'best',
    'max_depth': 3,
    'min_samples_split': 10,
    'max_features': None
}

# 3. Retrain with ALL training data (no CV)
model_final = DecisionTreeRegressor(**best_params, random_state=42)
model_final.fit(X_train, y_train)
y_pred = model_final.predict(X_test)

# 4. Calculate metrics
final_mse = mean_squared_error(y_test, y_pred)
final_rmse = sqrt(final_mse)
final_mae = mean_absolute_error(y_test, y_pred)

print(f" Exercise 1.3 Results (Model 3 Retrained)")
print(f"Final Test MSE:  {final_mse:.4f}")
print(f"Final Test RMSE: {final_rmse:.4f}")
print(f"Final Test MAE:  {final_mae:.4f}")
