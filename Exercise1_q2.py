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

# 2. Define the top 3 models from Exercise 1.1
top3_params = [
    # Model 1
    {'criterion': 'squared_error', 'splitter': 'random', 'max_depth': 5, 'min_samples_split': 10, 'max_features': None},
    # Model 2
    {'criterion': 'squared_error', 'splitter': 'random', 'max_depth': 5, 'min_samples_split': 5, 'max_features': None},
    # Model 3
    {'criterion': 'squared_error', 'splitter': 'best', 'max_depth': 3, 'min_samples_split': 10, 'max_features': None}
]

print("\nExercise 1.2: Test Data Results ")
test_results = []

for i, params in enumerate(top3_params):
    model = DecisionTreeRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    test_results.append(
        {'Model': f'Model {i + 1}', 'Test_MSE': round(mse, 4), 'Test_RMSE': round(rmse, 4), 'Test_MAE': round(mae, 4)})

df_test = pd.DataFrame(test_results)
print(df_test.to_string(index=False))

