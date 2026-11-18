import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


train_df = pd.read_csv('../data/training_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

features = ['x', 'y', 'Angular_velocity']
X_train = train_df[features]
y_train = train_df['Action']

X_test = test_df[features]
y_test = test_df['Action']


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test['x'], X_test['y'], X_test['Angular_velocity'],
           c='blue', s=10, alpha=0.3, label='Test Data (True States)')

ax.scatter(X_test['x'], X_test['y'], X_test['Angular_velocity'],
           c='red', s=10, alpha=0.3, label='Predicted States')

ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_zlabel('Angular Velocity')
ax.set_title('3D State Space Visualization')
ax.legend()
plt.show()
