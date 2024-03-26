import numpy as np
import torch

try:
    from envs.base_env import BaseEnv
except:
    from base_env import BaseEnv

def generate_random_negative_definite_matrix(n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    B = np.random.randn(n, n)

    # Compute A = B^T B (positive definite)
    A = B.T @ B
    C = -A
    return C

def sample_multi_price_env(dim, H, var, n_products=2, lower_price=5, upper_price=20, test=False):
    beta = generate_random_negative_definite_matrix(n_products)
    alpha = np.random.rand(n_products) * 100
    env = MultiPricesEnv(alpha, beta, dim, H, n_products, lower_price, upper_price, var=var)
    return env

class MultiPricesEnv(BaseEnv):
    def __init__(self, alpha, beta, dim, H, n_products, lower_price, upper_price, var=0.0):
        self.alpha = alpha
        self.beta = beta
        self.num_products = n_products
        self.dim = dim
        self.price_grid = np.linspace(lower_price, upper_price, dim)
        self.action_space = np.array([self.price_grid]*self.num_products)
        self.demands = np.zeros((self.num_products, dim))

        self.price_grid = np.linspace(lower_price, upper_price, dim)
        self.action_space = np.array([self.price_grid]*self.num_products)
        self.demands = np.zeros((self.num_products, dim, dim))
        self.means = np.zeros((self.num_products, dim, dim))

        for i in range(self.num_products):
            for j in range(dim):
                for k in range(dim):
                    self.demands[i, j, k] = self.alpha[i]
                    for l in range(self.num_products):
                        if l == i:
                            self.demands[i, j, k] += self.beta[i, l] * self.price_grid[j]
                        else:
                            self.demands[i, j, k] += self.beta[i, l] * self.price_grid[k]
                    self.means[i, j, k] = self.price_grid[j] * self.demands[i, j, k]

        self.total_means = np.sum(self.means, axis=0)
        self.opt_a_index = np.unravel_index(np.argmax(self.total_means, axis=None), self.total_means.shape)
        self.opt_a = np.zeros(self.total_means.shape)
        self.opt_a[self.opt_a_index] = 1.0
        self.opt_r = np.max(self.total_means)
        self.var = var
        self.H_context = H
        self.H = 1
        self.current_step = 0
        self.normalization_factor = np.sqrt(np.sum(alpha**2) + np.sum(beta**2))


    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0

    def transit(self, u):
        a = np.argmax(u)
        pt = self.price_grid[a]
        # REWARD FUNCTION
        r = (self.alpha + pt * self.beta + np.random.randn() * self.var) / self.normalization_factor
        return r

    def step(self, action):
        r = self.transit(action)
        self.current_step += 1
        done = (self.current_step >= self.H)
        return r, done, {}

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res


class MultiPricesEnvVec(BaseEnv):
    """
    Vectorized prices environment.
    """
    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.du = envs[0].dim

    def reset(self):
        return [env.reset() for env in self._envs]

    def step(self, actions):
        rews, dones = [], []
        for action, env in zip(actions, self._envs):
            rew, done, _ = env.step(action)
            rews.append(rew)
            dones.append(done)
        return rews, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    def deploy_eval(self, ctrl):
        # No variance during evaluation
        tmp = [env.var for env in self._envs]
        for env in self._envs:
            env.var = 0.0
        res = self.deploy(ctrl)
        for env, var in zip(self._envs, tmp):
            env.var = var
        return res

    def deploy(self, ctrl):
        # Print if the controller is of class BanditTransformerController
        us = []
        rs = []
        done = False

        if 'BanditTransformerController' in str(ctrl.__class__):
            name = 'BanditTransformerController'
        elif 'ThompsonSamplingPolicy' in str(ctrl.__class__):
            name = 'ThompsonSamplingPolicy'
        if 'OptPolicy' in str(ctrl.__class__):
            name = 'OptPolicy'
        elif 'UCB' in str(ctrl.__class__):
            name = 'UCB'

        price_grid = ctrl.envs[0].price_grid

        while not done:
            u = ctrl.act_numpy_vec()
            env = ctrl.envs[-1]
            print(env.alpha, env.beta)
            r, done, _ = self.step(u)

            done = all(done)
            us.append(u)
            rs.append(r)

        print(name)
        print(us[-1][-1])
        print(rs[-1][-1])
        print()
        us = np.concatenate(us)
        rs = np.concatenate(rs)
        return us, rs

