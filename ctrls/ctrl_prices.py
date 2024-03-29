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




class OptPolicy(Controller):
    def __init__(self, env, batch_size=1):
        super().__init__()
        self.env = env
        self.envs = self.env
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        return self.env.opt_a

    def act_numpy_vec(self):
        opt_as = [ env.opt_a for env in self.env ]
        return np.stack(opt_as, axis=0)
    

class ParaThompsonSamplingPolicy(Controller):
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


class GreedyOptPolicy(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def reset(self):
        return

    def act(self, x):
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()
        i = np.argmax(rewards)
        a = self.batch['context_actions'].cpu().detach().numpy()[0][i]
        self.a = a
        return self.a


class EmpMeanPolicy(Controller):
    def __init__(self, env, online=False, batch_size = 1):
        super().__init__()
        self.env = env
        self.online = online
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        i = np.argmax(b_mean)
        j = np.argmin(counts)
        if self.online and counts[j] == 0:
            i = j
        
        a = np.zeros(self.env.dim)
        a[i] = 1.0

        self.a = a
        return self.a

    def act_numpy_vec(self):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        i = np.argmax(b_mean, axis=-1)
        j = np.argmin(counts, axis=-1)
        if self.online:
            mask = (counts[np.arange(self.batch_size), j] == 0)
            i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0

        self.a = a
        return self.a


class PessMeanPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        pens = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean - pens

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a


    def act_numpy_vec(self):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean - bons

        i = np.argmax(bounds, axis=-1)
        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a


class UCBPolicy(Controller):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__()
        self.env = env
        self.envs = [self.env]
        self.const = const
        self.batch_size = batch_size

    def reset(self):
        return

    def act(self, x):
        actions = self.batch['context_actions'].cpu().detach().numpy()[0]
        rewards = self.batch['context_rewards'].cpu().detach().numpy().flatten()

        b = np.zeros(self.env.dim)
        counts = np.zeros(self.env.dim)
        for i in range(len(actions)):
            c = np.argmax(actions[i])
            b[c] += rewards[i]
            counts[c] += 1

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds)
        a = np.zeros(self.env.dim)
        a[i] = 1.0
        self.a = a
        return self.a

    def act_numpy_vec(self):
        actions = self.batch['context_actions']
        rewards = self.batch['context_rewards']

        b = np.zeros((self.batch_size, self.env.dim))
        counts = np.zeros((self.batch_size, self.env.dim))
        action_indices = np.argmax(actions, axis=-1)
        for idx in range(self.batch_size):
            actions_idx = action_indices[idx]
            rewards_idx = rewards[idx]
            for c in range(self.env.dim):
                arm_rewards = rewards_idx[actions_idx == c]
                b[idx, c] = np.sum(arm_rewards)
                counts[idx, c] = len(arm_rewards)

        b_mean = b / np.maximum(1, counts)

        # compute the square root of the counts but clip so it's at least one
        bons = self.const / np.maximum(1, np.sqrt(counts))
        bounds = b_mean + bons

        i = np.argmax(bounds, axis=-1)
        j = np.argmin(counts, axis=-1)
        mask = (counts[np.arange(counts.shape[0]), j] == 0)
        i[mask] = j[mask]

        a = np.zeros((self.batch_size, self.env.dim))
        a[np.arange(self.batch_size), i] = 1.0
        self.a = a
        return self.a


class LinUCBPolicy(OptPolicy):
    def __init__(self, env, const=1.0, batch_size=1):
        super().__init__(env)
        self.envs = [self.env]
        self.rand = True
        self.const = const
        self.arms = np.column_stack((np.ones(len(env.price_grid)), env.price_grid))
        self.d = self.arms.shape[1]
        self.dim = env.dim
        self.theta = np.zeros(self.d)
        self.init_cov = 1.0 * np.eye(self.d)
        self.batch_size = batch_size

    def act(self, x):
        if len(self.batch['rollin_rs'][0]) < 1:
            i = np.random.choice(np.arange(self.dim))
            hot_vector = np.zeros(self.dim)
            hot_vector[i] = 1
            return hot_vector

        else:
            actions = self.batch['rollin_us'].cpu().detach().numpy()[0]
            rewards = self.batch['rollin_rs'].cpu().detach().numpy().flatten()

            actions_indices = np.argmax(actions, axis=1)
            actions_arms = self.arms[actions_indices]

            cov = self.init_cov + actions_arms.T @ actions_arms
            cov_inv = np.linalg.inv(cov)

            theta = cov_inv @ actions_arms.T @ rewards

            best_arm_index = None
            best_value = -np.inf
            for i, arm in enumerate(self.arms):
                value = theta @ arm + self.const * np.sqrt(arm @ cov_inv @ arm)
                if value > best_value:
                    best_value = value
                    best_arm_index = i

            hot_vector = np.zeros(self.dim)
            hot_vector[best_arm_index] = 1
            self.a = hot_vector
            return hot_vector

    def act_numpy_vec(self):
        # TODO: parallelize this later
        actions_batch = self.batch['context_actions']
        rewards_batch = self.batch['context_rewards']

        if len(rewards_batch[0]) < 1:
            indices = np.random.choice(np.arange(self.dim), size=self.batch_size)
            hot_vectors = np.zeros((self.batch_size, self.dim))
            hot_vectors[np.arange(self.batch_size), indices] = 1
            return hot_vectors

        hot_vectors = []
        for i in range(self.batch_size):
            actions = actions_batch[i]
            rewards = rewards_batch[i]
            
            actions_indices = np.argmax(actions, axis=1)
            actions_arms = self.arms[actions_indices]

            cov = self.init_cov + actions_arms.T @ actions_arms
            cov_inv = np.linalg.inv(cov)

            theta = cov_inv @ actions_arms.T @ rewards
            theta = theta.flatten()

            best_arm_index = None
            best_value = -np.inf
            for i, arm in enumerate(self.arms):
                value = theta @ arm + self.const * np.sqrt(arm @ cov_inv @ arm)
                if value > best_value:
                    best_value = value
                    best_arm_index = i

            hot_vector = np.zeros(self.dim)
            hot_vector[best_arm_index] = 1
            hot_vectors.append(hot_vector)

        return np.array(hot_vectors)

