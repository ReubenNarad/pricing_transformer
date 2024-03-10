# For testing and inspecting

# from net import Transformer
# import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('datasets/trajs_prices_envs20_H500_d10_var0.3_eval.pkl', 'rb') as f:
    data = pickle.load(f)

a = data[5]

for i in range(min(len(a['context_actions']), 100)):
    print()
    print('means   :', [round(j, 3) for j in a['means']])
    print('actions:', [round(j, 3) for j in a['context_actions'][i]])
    print('reward:', round(a['context_rewards'][i], 3))
    print('cum_regret:', round(a['regrets'][i], 3))
    print('theta:', [round(a['thetas'][i][0], 3), round(a['thetas'][i][1], 3)])
print('true:', [round(a['alpha'], 3), round(a['beta'], 3)])

regrets = [np.mean([b['regrets'][i] for b in data]) for i in range(len(data[0]['regrets']))]

# Plot mean regret
plt.figure()
plt.plot(regrets)
plt.xlabel('Time Step')
plt.ylabel('Mean Regret')
plt.savefig('mean_regret.png')
plt.close()

for i in range(5):
    a = data[i]
    thetas_0 = [a['thetas'][j][0] for j in range(len(a['thetas']))]
    thetas_1 = [a['thetas'][j][1] for j in range(len(a['thetas']))]
    alpha = a['alpha']
    beta = a['beta']

    plt.figure()
    plt.plot(thetas_0, label='Theta 0')
    plt.plot(thetas_1, label='Theta 1')
    plt.plot([alpha] * len(a['thetas']), label='Alpha')
    plt.plot([beta] * len(a['thetas']), label='Beta')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'trajectory_{i}.png')
    plt.close()


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