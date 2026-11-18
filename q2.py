import pandas as pd
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

# 3. Define the Best Parameters from 3.1
best_params = {
    'hidden_layer_sizes': (100,),
    'activation': 'tanh',
    'solver': 'sgd',
    'learning_rate_init': 0.01
}

# 4. Retrain with early_stopping=False
model_3_2 = MLPRegressor(
    **best_params,
    early_stopping=False, # <--- 关键改动：关闭早停
    max_iter=1000,        # 为了让它在没有早停的情况下充分收敛
    random_state=42
)

print("Retraining the best model with early_stopping=False...")
model_3_2.fit(X_train_scaled, y_train)

# 5. Predict and Evaluate
y_pred = model_3_2.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\nExercise 3.2 Results ")
print(f"Retrained Test MSE:  {mse:.4f}")
print(f"Retrained Test RMSE: {rmse:.4f}")
print(f"Retrained Test MAE:  {mae:.4f}")
