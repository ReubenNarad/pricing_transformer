# For testing and inspecting

# from net import Transformer
# import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('metas/H_201_dim_10_meta.pkl', 'rb') as f:
    data = pickle.load(f)


for traj in range(11):

    opt = data['opt']['context_actions'][traj][:1]
    transformer = data['Transformer']['context_actions'][traj][:30]

    # Get the index of the optimal action
    optimal_action_index = np.argmax(opt[0])

    # Plot the trajectory as a heatmap
    plt.figure(figsize=(4, 9))  # Set the figsize to be more vertical
    plt.imshow(transformer, cmap='hot', aspect='auto')
    plt.title(f'Trajectory #{traj} (Optimal Action: {optimal_action_index})')
    plt.ylabel('Time Step')
    plt.xlabel('Action')
    plt.savefig(f'metas/traj_{traj}.png')