"""Overlay train and test mean reward for IDAAC, IDAAC+PLR, and PLR, capped at 5M steps."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_STEPS = 5_000_000

runs = {
    "IDAAC":      os.path.join(ROOT, "IDAAC_logs.csv"),
    "IDAAC + PLR": os.path.join(ROOT, "IDAAC+PLR_logs.csv"),
    "PLR (PPO)":  os.path.join(ROOT, "PLR_logs.csv"),
}

colors = {"IDAAC": "#1f77b4", "IDAAC + PLR": "#ff7f0e", "PLR (PPO)": "#2ca02c"}

def smooth(y, window=5):
    """Simple rolling mean for readability."""
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

for label, path in runs.items():
    df = pd.read_csv(path)
    df = df[df["train/total_num_steps"] <= MAX_STEPS]
    steps = df["train/total_num_steps"].values / 1e6  # in millions

    axes[0].plot(steps, smooth(df["train/mean_episode_reward"].values),
                 label=label, color=colors[label], linewidth=1.5)
    axes[1].plot(steps, smooth(df["test/mean_episode_reward"].values),
                 label=label, color=colors[label], linewidth=1.5)

for ax, title in zip(axes, ["Train Mean Reward", "Test Mean Reward"]):
    ax.set_xlabel("Environment Steps (millions)")
    ax.set_ylabel("Mean Reward")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(0, MAX_STEPS / 1e6)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = os.path.join(ROOT, "comparison_plot.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved to {out}")
