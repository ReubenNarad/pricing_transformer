import numpy as np
import torch

try:
    from envs.base_env import BaseEnv
except:
    from base_env import BaseEnv

def sample_price_env(dim, H, var):
    alpha = np.random.randint(20,80) / 10
    beta = np.random.randint(50,150) / -100 
    env = PricesEnv(alpha, beta, dim, H, var=var)
    return env

class PricesEnv(BaseEnv):
    def __init__(self, alpha, beta, dim, H, var=0.0, type='uniform'):
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.price_grid = np.linspace(1,5,self.dim)        
        
        self.demands = alpha + beta * self.price_grid
        self.means = self.demands * self.price_grid 
        
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.means.shape)
        self.opt_a[self.opt_a_index] = 1.0
        
        self.opt_r = np.max(self.means)
        
        self.dim = len(self.means)
        self.var = var
        self.du = self.dim

        # some naming issue here
        self.H_context = H
        self.H = 1
        
        self.current_step = 0

    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0

    def transit(self, u):
        a = np.argmax(u)
        pt = self.price_grid[a]
        # REWARD FUNCTION
        r = self.alpha + pt * self.beta + np.random.randn() * self.var
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


class PricesEnvVec(BaseEnv):
    """
    Vectorized prices environment.
    """
    def __init__(self, envs):
        self._envs = envs
        self._num_envs = len(envs)
        self.du = envs[0].du

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
        us = np.concatenate(us)
        rs = np.concatenate(rs)
        return us, rs

