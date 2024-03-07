import argparse
import os
import pickle
import random

import numpy as np
from skimage.transform import resize
from IPython import embed

import common_args
from envs import bandit_env, prices_env
from evals import eval_bandit
from utils import build_bandit_data_filename

from tqdm import tqdm
from scipy.stats import multivariate_normal

def rollin_prices(env, cov, orig=False, thompson=False, verbose=True):
    H = env.H_context
    opt_a_index = env.opt_a_index
    
    # For logging trajectories
    us, rs, regrets = [], [], []
    
    if thompson:
        theta = np.array([5, -1])
        cov = np.eye(2)
        rtxt = np.zeros(2)
        prices = []
        rewards = []
        sigma = .05
        cum_regret = 0
    
    for t in range(H):
        u = np.zeros(env.dim)
        
        if thompson:
            theta_draw = multivariate_normal(theta, np.linalg.inv(cov)).rvs()
            r_hat = [(theta_draw[0] + (price * theta_draw[1])) * price for price in env.price_grid]
            i = np.argmax(r_hat)
            u[i] = 1.0 
        else:
            i = np.random.choice(np.arange(env.dim))
            u[i] = 1.0 

        r = env.transit(u)
        
        us.append(u)
        rs.append(r)
        
        # Update TS
        if thompson:
            pt = env.price_grid[i]
            prices.append(pt)
            
            rewards.append(r)
            cum_regret += env.opt_r - (r * pt)
            regrets.append(cum_regret)
        if verbose:
            print()
            print(f"act: {u}")
            # print(f"means: {[round(num, 3) for num in env.means]}")
            print(f"opt: {env.opt_a}")
            print(f"opt_r: {round(env.opt_r, 3)}, act_r: {round(r * pt, 3)}, cum_regret: {round(cum_regret, 3)}")
            print(f"a_hat: {round(theta[0], 3)}, a:{round(env.alpha, 3)}")
            print(f"b_hat: {round(theta[1], 3)}, b:{round(env.beta, 3)}")
            xt = np.array([1,pt])
            cov += np.outer(xt,xt)
            rtxt += r*xt
            theta = np.linalg.inv(cov)@rtxt  

    us, rs = np.array(us), np.array(rs)
    return us, rs, regrets

def generate_prices_histories_from_envs(envs, n_hists, n_samples, cov, type, thompson):
    trajs = []
    for env in tqdm(envs):
        for j in range(n_hists):
            (
                context_actions,
                context_rewards,
                regrets
            ) = rollin_prices(env, cov=cov, thompson=thompson)
            for k in range(n_samples):
                optimal_action = env.opt_a

                traj = {
                    'context_actions': context_actions,
                    'context_rewards': context_rewards,
                    'optimal_action': optimal_action,
                    'regrets': regrets,
                    'prices': env.price_grid,
                    'means': env.means,
                    'demands': env.demands,
                    'alpha': env.alpha,
                    'beta': env.beta,
                }
                trajs.append(traj)
    return trajs

def generate_prices_histories(n_envs, dim, horizon, var, **kwargs):
    envs = [prices_env.sample_price_env(dim, horizon, var)
            for _ in range(n_envs)]
    print(envs[0])
    trajs = generate_prices_histories_from_envs(envs, thompson=True, **kwargs)
    return trajs


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    cov = args['cov']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']
    lin_d = args['lin_d']


    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }
    
    # Generate trajectories
    if env == 'prices':
        config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})

        train_trajs = generate_prices_histories(n_train_envs, **config)
        test_trajs = generate_prices_histories(n_test_envs, **config)
        eval_trajs = generate_prices_histories(n_eval_envs, **config)
                        
        train_filepath = build_bandit_data_filename(env, n_envs, config, mode=0)
        test_filepath = build_bandit_data_filename(env, n_envs, config, mode=1)
        eval_filepath = build_bandit_data_filename(env, n_eval_envs, config, mode=2)
    else:
        raise NotImplementedError


    # Save to /datasets
    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")
