import numpy as np
import torch
import random

try:
    from envs.base_env import BaseEnv
except:
    from base_env import BaseEnv

def sample_price_env(dim, H, var, lower_price=1, upper_price=10, test=False):
    '''
    Creates an instance of the PricesEnv class with the specified parameters.

    Parameters:
    - dim (int): The dimension of the environment.
    - H (float): The value of H parameter.
    - var (float): The value of var parameter.
    - lower_price (int, optional): The lower bound of the price range. Defaults to 1.
    - upper_price (int, optional): The upper bound of the price range. Defaults to 10.
    - test (bool, optional): Flag indicating whether to add noise to alpha and beta. Defaults to False.

    Returns:
    - env (PricesEnv): An instance of the PricesEnv class.'''

    # choice = np.random.randint(10)

    # if not test:
    #     alphas = [13., 30., 1., 4., 7., 4.2, 7., .52, 2., .25]
    #     betas = [-5., -8., -.17, -.5, -.7, -.35, -.5, -.032, -.11, -.012]
    # else:
    #     alphas = [30., 10., 6.5, 5., 3.5, 6., 3., 1.9, 1.6, 1.2]
    #     betas = [-14.5, -1.6, -.67, -.4, -.2, -1.5, -.36, -.14, -.1, -.06]
    
    # alpha = alphas[choice]
    # beta = betas[choice]

    # # Add noise to diversify parameters
    # alpha += np.random.normal(0, .1)
    # beta += np.random.normal(0, 0.01)

    # Draw envs in terms of price and reward
    opt_p = np.random.uniform(lower_price, upper_price)
    opt_r = np.random.uniform(5, 10)

    if test:
        opt_r = np.random.normal(loc=5, scale=3)
    else:
        opt_r = np.random.normal(loc=5, scale=3) 
    
    print(f"opt r: {opt_r}")

    alpha = 2 * opt_r / opt_p
    beta = - opt_r / opt_p ** 2
    
    env = PricesEnv(alpha, beta, dim, H, var=var, 
                    lower_price=lower_price, upper_price=upper_price)

    return env

class PricesEnv(BaseEnv):
    def __init__(self, alpha, beta, dim, H, lower_price, upper_price, var=0.0):
        self.alpha = alpha
        self.beta = beta
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
        r = self.alpha + pt * self.beta  + np.random.randn() * self.var
        
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

