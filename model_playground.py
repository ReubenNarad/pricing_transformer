from net import Transformer
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
# For dissecting the model

with open('datasets/trajs_prices_envs100_hists1_samples1_H500_d10_var0.3_cov0.0_train.pkl', 'rb') as f:
    data = pickle.load(f)

a = data[2]

for i in range(min(len(a['context_actions']), 10)):
    print()
    print('means   :', [round(j, 3) for j in a['means']])
    print('actions:', a['context_actions'][i])
    print('reward:', a['context_rewards'][i])
print('true:', [a['alpha'], a['beta']])

mean_regrets = np.mean([d['regrets'] for d in data], axis=0)

# Plot mean regrets
plt.plot(mean_regrets)
plt.savefig('img.png')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config = {
#     'horizon': 50,
#     'state_dim': 1,
#     'action_dim': 10,
#     'n_layer': 4,
#     'n_embd': 32,
#     'n_head': 4,
#     'dropout': 0,
#     'test': True,
# }

# model = Transformer(config).to(device)

# model_path = 'models/prices_shufTrue_lr0.0001_do0_embd32_layer4_head4_envs100_hists1_samples1_var0.3_cov0.0_H50_d10_seed1_epoch100.pt'


# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint)
# model.eval()

# print(a)

# model(a)