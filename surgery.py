import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from net import *
from dataset import Dataset
from utils import build_prices_data_filename

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = {
    'horizon': 200,
    'state_dim': 2,
    'action_dim': 30,
    'n_layer': 2,
    'n_embd': 32,
    'n_head': 1,
    'dropout': False,
    'lr': 0.0001,
    'dim': 30,
    'seed': 2,
    'test': False,
    'store_gpu': True,
    'n_envs': 50000,
    'var': 0.0,
}

model = Transformer(cfg)

print(model)

model.load_state_dict(torch.load('models/model_d30_envs50000_H200_head1_layer1_lr0.001_seed2_epoch200.pt'))
model = model.to(device)

# Define the hook function
activations = {}

def hook_fn(module, input, output):
    activations['attn'] = output

# Register the hook
head_0 = model.transformer.h[0].attn.c_attn
hook_handle = head_0.register_forward_hook(hook_fn)


# Build the dataset filename
filename = build_prices_data_filename(cfg['n_envs'], cfg, mode=2)

filename = "datasets/trajs_d30_H200_envs100_var0.0_eval.pkl"

# Load the dataset
dataset_config = {
    'shuffle' : False,
    'action_dim': cfg['action_dim'],
    'horizon': cfg['horizon'],
    'dim': cfg['dim'],
    'var': cfg['var'],
    'store_gpu': True,
    'truncate': False
}

params = {
    'batch_size': 1,
    'shuffle': False,
}

test_dataset = Dataset(filename, dataset_config)
test_loader = torch.utils.data.DataLoader(test_dataset, **params, )

# Print one batch
for i, data in enumerate(test_loader):
    if i == 0:
        print("Batch: ", data['context_actions'].shape)
        output = model(data)
        break

# Extract the activations
KQV = activations['attn'][0, :, :]

# Split into K, Q, and V
K, Q, V = torch.split(KQV, cfg['n_embd'], dim=-1)  # Each will have shape [101, 64]

# Transpose K for dot product
K_transposed = K.transpose(-1, -2)  # Shape: [64, 101]

# Compute self-attention scores
attention_scores = torch.matmul(Q, K_transposed)  # Shape: [101, 101]

# Optionally, you can scale the attention scores by the square root of the dimension
d_k = cfg['n_embd']
attention_scores = attention_scores / (d_k ** 0.5)

# Convert to numpy for visualization
attention_scores_np = attention_scores.detach().cpu().numpy()

# attention_scores_np = np.triu(attention_scores_np)

# Plot the attention scores
plt.figure(figsize=(10, 8))
sns.heatmap(attention_scores_np, cmap='viridis', mask=np.isnan(attention_scores_np))
plt.title('Self-Attention Scores (Lower Triangular)')
plt.xlabel('Key Positions')
plt.ylabel('Query Positions')
plt.savefig('attention_scores.png')