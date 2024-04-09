import numpy as np
import torch

try:
    from envs.base_env import BaseEnv
except:
    from base_env import BaseEnv

def sample_price_env(dim, H, var, opt_a_index=None, lower_price=5, upper_price=10, test=False):
    prices = np.linspace(lower_price, upper_price, dim)
    if False:
        # Draws envs with uniformly distributed optimal actions
        price = prices[opt_a_index]
        alpha = np.random.randint(55, 95) / 10
        beta = - alpha / (2 * price)
    else:
        # Draws envs with uniformly distributed alpha and beta
        alpha = np.random.randint(50,100) / 10
        beta = np.random.randint(50,100) / -100 

        # Draws envs with normally distributed alpha and beta
        # alpha = np.random.normal(0.5, 0.1)
        # beta = np.random.normal(-0.5, 0.1)
        
    env = PricesEnv(alpha, beta, dim, H, var=var, lower_price=lower_price, upper_price=upper_price)
    return env

class PricesEnv(BaseEnv):
    def __init__(self, alpha, beta, dim, H, lower_price, upper_price, var=0.0, type='uniform'):
        self.normalization_factor = np.sqrt(alpha**2 + beta**2)
        self.alpha = alpha/self.normalization_factor
        self.beta = beta/self.normalization_factor
        self.dim = dim
        self.price_grid = np.linspace(lower_price, upper_price, dim)   
        self.demands = alpha + beta * self.price_grid
        self.means = self.demands * self.price_grid 
        self.du = dim
        self.opt_a_index = np.argmax(self.means)
        self.opt_a = np.zeros(self.means.shape)
        self.opt_a[self.opt_a_index] = 1.0
        self.opt_price = self.price_grid[self.opt_a_index]
        self.opt_r = np.max(self.means)
        self.var = var

        # some naming issue here
        self.H_context = H
        #FIXME what is this??
        self.H = 1
        self.current_step = 0       

    def get_arm_value(self, u):
        return np.sum(self.means * u)

    def reset(self):
        self.current_step = 0


    def step(self, action):
        """
        Takes an action and computes the reward.

        Parameters:
        - u: A one-hot numpy array representing the action.

        Returns:
        - r: The calculated reward.

        """
        a = np.argmax(action)
        pt = self.price_grid[a]
        # REWARD FUNCTION
        r = (self.alpha + pt * self.beta ) / self.normalization_factor + np.random.randn() * self.var
        
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
        us = []
        rs = []
        done = False
        while not done:
            # calls the model on the batch
            # returns env x actions (one hot)
            u = ctrl.act_numpy_vec()
            # takes the action and returns the reward
            r, done, _ = self.step(u)
            done = all(done)
            us.append(u)
            rs.append(r)
        us = np.concatenate(us)
        rs = np.concatenate(rs)
        
        return us, rs

