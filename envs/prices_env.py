import numpy as np
import torch

try:
    from envs.base_env import BaseEnv
except:
    from base_env import BaseEnv

def sample_price_env(dim, H, var):
    alpha = np.random.randint(20,110) / 10
    beta = np.random.randint(50,150) / -100
    env = PricesEnv(alpha, beta, dim, H, var=var)
    return env

class PricesEnv(BaseEnv):
    def __init__(self, alpha, beta, dim, H, var=0.0, type='uniform'):
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.price_grid = np.linspace(1,5,self.dim)        
        
        self.demands = np.maximum(alpha + beta * self.price_grid, 0)
        self.means = self.demands * self.price_grid 
        
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.means.shape)
        self.opt_a[self.opt_a_index] = 1.0
        
        self.opt_r = np.max(self.means)
        
        self.dim = len(self.means)
        self.var = var
        self.dx = 1
        self.du = self.dim

        # some naming issue here
        self.H_context = H
        self.H = 1

    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0

    def transit(self, u):
        a = np.argmax(u)
        
        # REWARD FUNCTION
        r = max(self.demands[a] + np.random.normal(0, self.var), 0)
        return r

    def step(self, action):
        if self.current_step >= self.H:
            raise ValueError("Episode has already ended")

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
