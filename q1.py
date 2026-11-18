import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
train_data = pd.read_csv("../data/training_data.csv")
test_data = pd.read_csv("../data/test_data.csv")

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 2. Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Parameter combinations
hidden_layers = [(50,), (100,), (50, 50)]
activations = ['relu', 'tanh']
solvers = ['adam', 'sgd']
learning_rates = [0.001, 0.01]

results = []

for hl in hidden_layers:
    for act in activations:
        for sol in solvers:
            for lr in learning_rates:

                model = MLPRegressor(
                    hidden_layer_sizes=hl,
                    activation=act,
                    solver=sol,
                    learning_rate_init=lr,
                    early_stopping=True,
                    max_iter=1000,
                    random_state=42
                )

                # 直接全量训练（不交叉验证）
                model.fit(X_train_scaled, y_train)

                # 预测测试集
                y_pred = model.predict(X_test_scaled)

                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                results.append({
                    'hidden_layer_sizes': str(hl),
                    'activation': act,
                    'solver': sol,
                    'learning_rate': lr,
                    'Test_MSE': round(mse, 4),
                    'Test_RMSE': round(rmse, 4),
                    'Test_MAE': round(mae, 4)
                })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='Test_MSE', ascending=True)

print("\n Exercise 3.1: Neural Network Results ")
print(df_results.head(10).to_string(index=False))

df_results.to_csv("exercise3_results.csv", index=False)


