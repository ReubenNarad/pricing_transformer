import pickle
import numpy as np
import torch

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # load data from paths
        if not isinstance(path, list):
            path = [path]
        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_actions = []
        context_rewards = []
        optimal_actions = []

        # context_actions: (n_samples, horizon, dim)
        # context_rewards: (n_samples, horizon, 1)
        # optimal_actions: (n_samples, 1)
        for traj in self.trajs:
            context_actions.append(traj['context_actions'])
            context_rewards.append(traj['context_rewards'])
            optimal_actions.append(traj['optimal_action'])

        context_actions = np.array(context_actions)
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        optimal_actions = np.array(optimal_actions)

        self.dataset = {
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }

        self.zeros = np.zeros(
            config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['context_actions'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_actions': self.dataset['context_actions'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }

        # if self.shuffle:
        #     perm = torch.randperm(self.horizon)
        #     res['context_actions'] = res['context_actions'][perm]
        #     res['context_rewards'] = res['context_rewards'][perm]

        return res