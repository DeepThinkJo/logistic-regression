# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/apples_and_oranges.csv")
x = df[["Weight", "Size"]].to_numpy()
y = np.where(df["Class"].to_numpy()=="apple", 1, 0)

# Scale features and construct the design matrix with bias term
mu1, mu2 = np.mean(x, axis=0)
sigma1, sigma2 = np.std(x, axis=0)
x_scaled = (x - np.array([mu1, mu2])) / np.array([sigma1, sigma2])
X = np.hstack([np.ones((x_scaled.shape[0], 1)), x_scaled])

# Train the model using batch gradient descent
m = y.shape[0]
param = np.array([0, 0, 0], dtype=float)
param_history = []
loss_history = []
epochs = 10000
lr = 10 ** (-2)

# Set a constant to prevent numerical errors
eps = 1e-15

for epoch in range(epochs):
    pred = 1 / (1 + np.exp(-1.0 * X @ param))
    pred_clipped = np.clip(pred, eps, 1-eps)
    loss = - (y.T @ np.log(pred_clipped) + (1.0 - y).T @ np.log(1.0 - pred_clipped)) / m
    loss_history.append({"iteration": epoch, "loss": loss})
    param_history.append({"iteration": epoch, "param0": param[0], "param1": param[1], "param2": param[2]})
    param -= lr * (1.0 / m) * X.T @ (pred - y)

# Save loss history and parameter history to CSV files
loss_history_df = pd.DataFrame(loss_history)
param_history_df = pd.DataFrame(param_history)
loss_history_df.to_csv("../results/csv/loss_history_scaled.csv", index=False)
param_history_df.to_csv("../results/csv/param_history_scaled.csv", index=False)