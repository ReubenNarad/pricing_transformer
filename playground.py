# For testing and inspecting

# from net import Transformer
# import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('metas/H_100_dim_10_meta.pkl', 'rb') as f:
    data = pickle.load(f)


for traj in range(11):
    opt = data['opt']
    transformer = data['Transformer']['context_actions'][traj][-30:]
    thomp = data['ParamThomp']['context_actions'][traj][-30:]

    # Get the index of the optimal action
    optimal_action_index = np.argmax(opt[0])

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 9))

    # Plot the transformer on the first subplot
    axes[0].imshow(transformer, cmap='hot', aspect='auto')
    axes[0].set_title(f'Transformer Traj #{traj} (Optimal A: {optimal_action_index})')
    axes[0].set_ylabel('Time Step')
    axes[0].set_xlabel('Action')

    # Plot LinUCB on the second subplot
    axes[1].imshow(thomp, cmap='hot', aspect='auto')
    axes[1].set_title(f'P. Thomp. Traj #{traj} (Optimal A: {optimal_action_index})')
    axes[1].set_ylabel('Time Step')
    axes[1].set_xlabel('Action')

    # Adjust layout so titles don't overlap
    plt.tight_layout()
    plt.savefig(f'metas/traj_{traj}.png')

    # Ensure the plots are not displayed in the console
    plt.close(fig)