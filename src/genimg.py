# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load dataset
df = pd.read_csv("../data/apples_and_oranges.csv")
x = df[["Weight", "Size"]].to_numpy()
y = np.where(df["Class"].to_numpy()=="apple", 1, 0)
mu1, mu2 = np.mean(x, axis=0)
sigma1, sigma2 = np.std(x, axis=0)

# Plot and save initial data scatter plot
apple_condition = (y == 1)
apple_dataset = x[apple_condition]
orange_dataset = x[~apple_condition]

fig, ax = plt.subplots(figsize = (7, 5))
ax.scatter(apple_dataset.T[0], apple_dataset.T[1], color="red", alpha=0.6, edgecolors="none", label="Apple")
ax.scatter(orange_dataset.T[0], orange_dataset.T[1], color="orange", alpha=0.6, edgecolors="none", label="Orange")
ax.set_title("Weight and Size of Apples and Oranges")
ax.set_xlabel("Weight (x1)")
ax.set_ylabel("Size (x2)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")
fig.savefig("../results/plots/data_scatter.png", dpi=300, bbox_inches="tight")

# Draw loss history plot (no scaling version)
loss_history = pd.read_csv("../results/csv/loss_history.csv")
fig, ax = plt.subplots(figsize = (7, 5))
ax.plot(loss_history["iteration"], loss_history["loss"], color="blue", linewidth=1)
ax.set_title("Loss History (Without Scaling)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.grid(True, linestyle="--", alpha=0.5)
fig.savefig("../results/plots/loss_history.png", dpi=300, bbox_inches="tight")

# Draw loss history plot (scaling version)
loss_history_scaled = pd.read_csv("../results/csv/loss_history_scaled.csv")
fig, ax = plt.subplots(figsize = (7, 5))
ax.plot(loss_history_scaled["iteration"], loss_history_scaled["loss"], color="blue", linewidth=2)
ax.set_title("Loss History (With Scaling)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.grid(True, linestyle="--", alpha=0.5)
fig.savefig("../results/plots/loss_history_scaled.png", dpi=300, bbox_inches="tight")

# Set parameters for the animation of the fitted line during iterations
num_frames = 100
interval = 100  # milliseconds

# Generate frames for the animation of the fitted line during iterations (no scaling version)
param_history_df = pd.read_csv("../results/csv/param_history.csv")
param_history = param_history_df[["param0", "param1", "param2"]].to_numpy()
param_iterations = param_history_df["iteration"].to_numpy()

x1_grid = np.linspace(x.T[0].min()-1, x.T[0].max()+1, 100)

theta_final = param_history[-1]

# Place more frames in the early iterations where the boundary moves quickly.
# Linear sampling concentrates almost every frame near the convergence endpoint
# and can leave only the first and final frames visible.
frame_positions = np.unique(
    np.floor(1 + (len(param_history) - 1) * np.linspace(0, 1, num_frames) ** 3).astype(int)
)
frame_positions[-1] = len(param_history) - 1
frames = param_history[frame_positions]
frame_indices = param_iterations[frame_positions]
frames = np.array(frames)

# Generate the animation of the fitted line during iterations (no scaling version)
fig, ax = plt.subplots(figsize = (7, 5))
ax.scatter(apple_dataset.T[0], apple_dataset.T[1],
           color="red", alpha=0.6, edgecolors="none", label="Apple")
ax.scatter(orange_dataset.T[0], orange_dataset.T[1],
           color="orange", alpha=0.6, edgecolors="none", label="Orange")
first_t0, first_t1, first_t2 = frames[0]   
y_vals = -1 / first_t2 * (first_t0 + first_t1 * x1_grid)
line, = ax.plot(x1_grid, y_vals,
                color="blue", linewidth=2, label="Decision Boundary")
ax.set_xlim(x.T[0].min()-1, x.T[0].max()+1)
ax.set_ylim(x.T[1].min()-0.1, x.T[1].max()+0.1)
ax.set_xlabel("Weight (x1)")
ax.set_ylabel("Size (x2)")
title_text = ax.set_title("")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

def init():
    line.set_data([], [])
    title_text.set_text("")
    return line, title_text

def update(frame_index):
    cur_param = frames[frame_index]
    t0, t1, t2 = cur_param[0], cur_param[1], cur_param[2]

    y_vals = -1 / t2 * (t0 + t1 * x1_grid)

    line.set_data(x1_grid, y_vals)
    title_text.set_text(f"Iteration {frame_indices[frame_index]} | {t0:.2f} + {t1:.2f}x1 + {t2:.2f}x2 = 0")
    return line, title_text

ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, interval=interval)
ani.save("../results/animations/boundary_fitting.gif", writer="pillow", fps=1000/interval)

# Generate frames for the animation of the fitted line during iterations (scaling version)
param_history_scaled_df = pd.read_csv("../results/csv/param_history_scaled.csv")
param_history_scaled = param_history_scaled_df[["param0", "param1", "param2"]].to_numpy()
param_iterations_scaled = param_history_scaled_df["iteration"].to_numpy()

x1_grid = np.linspace(x.T[0].min()-1, x.T[0].max()+1, 100)

frame_positions_scaled = np.unique(
    np.floor(1 + (len(param_history_scaled) - 1) * np.linspace(0, 1, num_frames) ** 3).astype(int)
)
frame_positions_scaled[-1] = len(param_history_scaled) - 1
frames_scaled = param_history_scaled[frame_positions_scaled]
frame_indices_scaled = param_iterations_scaled[frame_positions_scaled]
frames_scaled = np.array(frames_scaled)

# Generate the animation of the fitted line during iterations (scaling version)
fig, ax = plt.subplots(figsize = (7, 5))
ax.scatter(apple_dataset.T[0], apple_dataset.T[1],
           color="red", alpha=0.6, edgecolors="none", label="Apple")
ax.scatter(orange_dataset.T[0], orange_dataset.T[1],
           color="orange", alpha=0.6, edgecolors="none", label="Orange")
first_t0, first_t1, first_t2 = frames_scaled[0]
y_vals = -sigma2 / first_t2 * ((first_t0 - first_t1 * mu1 / sigma1 - first_t2 * mu2 / sigma2) + first_t1 / sigma1 * x1_grid)
line, = ax.plot(x1_grid, y_vals,
                color="blue", linewidth=2, label="Decision Boundary")
ax.set_xlim(x.T[0].min()-1, x.T[0].max()+1)
ax.set_ylim(x.T[1].min()-0.1, x.T[1].max()+0.1)
ax.set_xlabel("Weight (x1)")
ax.set_ylabel("Size (x2)")
title_text = ax.set_title("")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

def init():
    line.set_data([], [])
    title_text.set_text("")
    return line, title_text

def update(frame_index):
    cur_param = frames_scaled[frame_index]
    t0, t1, t2 = cur_param[0], cur_param[1], cur_param[2]

    s0 = t0 - t1 * mu1 / sigma1 - t2 * mu2 / sigma2
    s1 = t1 / sigma1
    s2 = t2 / sigma2

    y_vals = -(s0 + s1 * x1_grid) / s2 

    line.set_data(x1_grid, y_vals)
    title_text.set_text(f"Iteration {frame_indices_scaled[frame_index]} | {s0:.2f} + {s1:.2f}x1 + {s2:.2f}x2 = 0")
    return line, title_text

ani = FuncAnimation(fig, update, frames=len(frames_scaled), init_func=init, blit=True, interval=interval)
ani.save("../results/animations/boundary_fitting_scaled.gif", writer="pillow", fps=1000/interval)