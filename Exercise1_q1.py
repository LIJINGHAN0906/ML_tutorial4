import numpy as np
import gymnasium as gym
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("../data/training_data.csv")
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

dt_model = DecisionTreeRegressor(
    criterion='squared_error', splitter='best',
    max_depth=3, min_samples_split=10, max_features=None,
    random_state=42
)
dt_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

nn_model = MLPRegressor(
    hidden_layer_sizes=(100,), activation='tanh', solver='sgd',
    learning_rate_init=0.01, early_stopping=True, max_iter=1000, random_state=42
)
nn_model.fit(X_train_scaled, y_train)

env = gym.make('Pendulum-v1')

def test_agent(model, scaler=None, episodes=5):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:

            state_input = state.reshape(1, -1)
            if scaler is not None:
                state_input = scaler.transform(state_input)


            action = model.predict(state_input)

            action = np.clip(action, -2.0, 2.0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


print("Exercise 4.1: Deployment Performance ")
print("Testing agents in Pendulum-v1 (Higher reward is better, 5 episodes)...")

dt_reward = test_agent(dt_model)
print(f"Decision Tree Avg Reward: {dt_reward:.2f}")

lr_reward = test_agent(lr_model)
print(f"Linear Regression Avg Reward: {lr_reward:.2f}")

nn_reward = test_agent(nn_model, scaler=scaler)
print(f"Neural Network Avg Reward: {nn_reward:.2f}")

env.close()
