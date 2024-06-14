import os
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmaps(run_name):
    # Load pickled probs
    with open(f'runs/{run_name}/evals/logits.pkl', 'rb') as f:
        probs, opts = pickle.load(f)

    # Convert logits to probabilities
    probs = F.softmax(torch.tensor(probs), dim=-1)

    # Create dir 'heatmap' if it doesn't exist
    if not os.path.exists(f'runs/{run_name}/evals/heatmaps'):
        os.makedirs(f'runs/{run_name}/evals/heatmaps')

    # Create heatmap of actions
    for TRAJ in range(10):
        traj = probs[TRAJ, :, :]
        opt = opts[TRAJ]
        sns.heatmap(traj, cmap='viridis')
        plt.title(f'Optimal Price: {opt}')
        plt.ylabel('Trajectory step')
        plt.xlabel('Price index')
        plt.savefig(f'runs/{run_name}/evals/heatmaps/traj_{TRAJ}.png')
        plt.clf()
