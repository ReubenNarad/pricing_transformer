import pickle
import torch
import torch.nn.functional as F

# load pickled probs
with open('logits.pkl', 'rb') as f:
    probs = pickle.load(f)

# convert logits to probabilities
probs = F.softmax(torch.tensor(probs), dim=-1)

# Create heatmap of actions
import matplotlib.pyplot as plt
import seaborn as sns

probs = probs[0, :, :]
sns.heatmap(probs, cmap='viridis')
plt.savefig('heatmap.png')
