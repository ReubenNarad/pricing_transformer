import itertools

import numpy as np
import scipy
import torch
from scipy.stats import multivariate_normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Controller:
    def set_batch(self, batch):
        self.batch = batch

    def set_batch_numpy_vec(self, batch):
        self.set_batch(batch)

    def set_env(self, env):
        self.env = env

class ParaThompsonSamplingPolicy():
    def __init__(self, env, std=.1, theta_0=[0, 0], cov_0=np.eye(2), warm_start=False, batch_size=1):
        super().__init__()
        self.price_grid = env.price_grid
        self.std = std
        self.variance = std**2
        self.theta_0 = theta_0
        self.cov_0 = cov_0*self.variance
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        # Reset posteriors over alpha and beta
        self.thetas = np.zeros((self.batch_size, 2))
        self.covs = np.tile(self.cov_0, (self.batch_size, 1, 1))
        self.rtxts = [np.zeros(2) for _ in range(self.batch_size)]
        self.prices = [[] for _ in range(self.batch_size)] 
        self.rewards = [[] for _ in range(self.batch_size)]

    def set_batch_numpy_vec(self, batch):       
        self.reset()
        self.batch = batch
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards'][:, :, 0]
        self.envs = self.batch['envs']

        for idx in range(self.batch_size):
            self.update_posterior(idx, actions[idx], rewards[idx])

    def update_posterior(self, idx, actions, rewards):
        for t in range(len(actions)):
            a = np.argmax(actions[t])
            price = self.envs[idx].price_grid[a]
            reward = rewards[t]
            xt = np.array([1, price])
            self.covs[idx] += np.outer(xt, xt)
            self.rtxts[idx] += reward * xt
            self.thetas[idx] = np.linalg.inv(self.covs[idx]) @ self.rtxts[idx]

    def act_numpy_vec(self, a=1):
        actions = np.zeros((self.batch_size, self.envs[0].dim))
        for idx in range(self.batch_size):
            theta_draw = multivariate_normal(self.thetas[idx], a*np.linalg.inv(self.covs[idx])).rvs()
            r_hat = [(theta_draw[0] + (price * theta_draw[1])) * price for price in self.envs[idx].price_grid]            
            opt_a_index = np.argmax(r_hat)
            actions[idx, opt_a_index] = 1.0

        return actions


class BanditTransformerController(Controller):
    def __init__(self, model, sample=False,  batch_size=1):
        self.model = model
        self.du = model.config['action_dim']
        self.dx = model.config['state_dim']
        self.H = model.horizon
        self.sample = sample
        self.batch_size = batch_size
        self.zeros = torch.zeros(batch_size, self.dx**2 + self.du + 1).float().to(device)
        self.envs = []

    def set_batch_numpy_vec(self, batch):
        # Convert each element of the batch to a torch tensor
        new_batch = {}
        for key in batch.keys():
            if key != 'envs':
                new_batch[key] = torch.tensor(batch[key]).float().to(device)
            else:
                self.envs = batch[key]
        self.set_batch(new_batch)


    def act_numpy_vec(self, opt_as=None):
        self.batch['zeros'] = self.zeros

        a = self.model(self.batch)
        print(f"ACTION: {a}")
        a = a.cpu().detach().numpy()

        action_indices = np.argmax(a, axis=-1)
        print(f"OPT AS:\n{opt_as}")

        actions = np.zeros((self.batch_size, self.du))
        actions[np.arange(self.batch_size), action_indices] = 1.0

        print(f"ACTIONS:\n {actions}")
        return actions


