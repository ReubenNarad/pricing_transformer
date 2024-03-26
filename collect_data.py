import argparse
import os
import pickle
import random

import numpy as np

import common_args
from envs import prices_env, multi_prices_env
from utils import build_prices_data_filename

from tqdm import tqdm
from scipy.stats import multivariate_normal

def rollin_prices(env, orig=False, verbose=False):
    lambda_0 = .01
    H = env.H_context
    opt_a_index = env.opt_a_index
    
    # For logging trajectories
    us, rs, regrets, thetas = [], [], [], []
    
    for t in range(H):
        u = np.zeros(env.dim)
        
        i = np.random.choice(np.arange(env.dim))
        
        u[i] = 1.0
        r, done, _ = env.step(u)
        us.append(u)
        rs.append(r)
            
    us, rs = np.array(us), np.array(rs)
    return us, rs, regrets, thetas


def generate_prices_histories_from_envs(envs, n_samples):
    trajs = []
    for env in tqdm(envs):
        (
            context_actions,
            context_rewards,
            regrets,
            thetas,
        ) = rollin_prices(env)
        for k in range(n_samples):
            optimal_action = env.opt_a
            traj = {
                'context_actions': context_actions,
                'context_rewards': context_rewards,
                'optimal_action': optimal_action,
                'regrets': regrets,
                'prices': env.price_grid,
                'means': env.means,
                'thetas' : thetas,
                'alpha': env.alpha,
                'beta': env.beta,
            }
            trajs.append(traj)
    return trajs

def generate_prices_histories(n_envs, dim, horizon, var, **kwargs):
    envs = []
    for _ in range(int(n_envs / dim)):
        for a in range(dim):
            envs.append(prices_env.sample_price_env(dim, horizon, var, opt_a_index=a))
    trajs = generate_prices_histories_from_envs(envs, **kwargs)
    return trajs


# For multiproduct setting

def rollin_multi_prices(env, orig=False, verbose=False):
    print("Rollin multi prices not implemented yet.")
    return [], [], []

def generate_multi_prices_histories_from_envs(envs, n_samples):
    trajs = []
    for env in tqdm(envs):
        (
            context_actions,
            context_rewards,
            regrets
        ) = rollin_multi_prices(env)
        for k in range(n_samples):
            optimal_action = env.opt_a
            traj = {
                'context_actions': context_actions,
                'context_rewards': context_rewards,
                'optimal_action': optimal_action,
                'regrets': regrets,
                'prices': env.price_grid,
                'means': env.means,
                'alpha': env.alpha,
                'beta': env.beta,
            }
            trajs.append(traj)
    return trajs

def generate_multi_prices_histories(n_envs, dim, horizon, var, **kwargs):
    envs = []
    for _ in range(int(n_envs / dim)):
        for a in range(dim):
            envs.append(multi_prices_env.sample_multi_price_env(dim, horizon, var, opt_a_index=a))
    trajs = generate_multi_prices_histories_from_envs(envs, **kwargs)
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
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']
    lin_d = args['lin_d']


    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_samples': n_samples,
        'horizon': horizon,
    }
    
    # Generate trajectories
    config.update({'dim': dim, 'var': var})


    if env == 'prices':
        train_trajs = generate_prices_histories(n_train_envs, **config)
        test_trajs = generate_prices_histories(n_test_envs, **config)
        eval_trajs = generate_prices_histories(n_eval_envs, **config)

    if env == 'multi_prices':
        train_trajs = generate_multi_prices_histories(n_train_envs, **config)
        test_trajs = generate_multi_prices_histories(n_test_envs, **config)
        eval_trajs = generate_multi_prices_histories(n_eval_envs, **config)

    train_filepath = build_prices_data_filename(env, n_envs, config, mode=0)
    test_filepath = build_prices_data_filename(env, n_envs, config, mode=1)
    eval_filepath = build_prices_data_filename(env, n_eval_envs, config, mode=2)


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
