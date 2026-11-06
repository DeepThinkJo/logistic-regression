# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset and construct the design matrix with bias term
df = pd.read_csv("../data/apples_and_oranges.csv")
x = df[["Weight", "Size"]].to_numpy()
X = np.hstack([np.ones((x.shape[0], 1)), x])
y = np.where(df["Class"].to_numpy()=="apple", 1, 0)

# Train the model using batch gradient descent
m = y.shape[0]
param = np.array([0, 0, 0], dtype=float)
param_history = []
loss_history = []
epochs = 1000000
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
param_gd = param.copy()

# Save loss history and parameter history to CSV files
loss_history_df = pd.DataFrame(loss_history)
param_history_df = pd.DataFrame(param_history)
loss_history_df.to_csv("../results/csv/loss_history.csv", index=False)
param_history_df.to_csv("../results/csv/param_history.csv", index=False)