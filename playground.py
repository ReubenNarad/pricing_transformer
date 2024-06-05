import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# load pickled probs
with open('logits.pkl', 'rb') as f:
    probs, opts = pickle.load(f)

# convert logits to probabilities
probs = F.softmax(torch.tensor(probs), dim=-1)

# Create heatmap of actions

TRAJ = next((opts.index(opt) for opt in [15, 17, 18, 19] if opt in opts), 0)

probs = probs[TRAJ, :30, :]
opt = opts[TRAJ]
sns.heatmap(probs, cmap='viridis')
plt.title(f'Optimal Price: {opt}')
plt.ylabel('Trajectory step')
plt.xlabel('Price index')
plt.savefig('heatmap.png')
