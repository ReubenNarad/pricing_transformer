import os
import pickle

import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import common_args
from envs import prices_env
from utils import build_prices_data_filename


def generate_uniform_trajectory(env): 
    '''
    Create a trajectory of actions and rewards for a given environment. We are doing uniform.

    Input: 
    env: PricesEnv object

    Outputs:
    us: H x env.dim actions as one hot encoded vectors
    rs: H x 1 rewards
    '''
    
    H = env.H_context
    
    # For logging trajectories
    us, rs = [], []
    
    for t in range(H):
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim))
        u[i] = 1.0
        r, _, _ = env.step(u)
        us.append(u)
        rs.append(r)
        
            
    us, rs = np.array(us), np.array(rs)
    return us, rs


def generate_prices_histories(n_envs, dim, horizon, var, env_type='prices', test=False, **kwargs):
    """
    Generate price histories for multiple environments.

    Args:
        n_envs (int): The total number of environments to generate.
        dim (int): The dimension of each environment.
        horizon (int): The time horizon for each price history.
        var (float): The variance of the price distribution.
        env_type (str): The type of environment to generate. Default is 'prices'.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of price histories generated for the environments.

    """
    envs = []
    for _ in range(n_envs):
        envs.append(prices_env.sample_price_env(dim, horizon, var, test=test))
   

    trajs = []
    for env in tqdm(envs):
        (
            context_actions,
            context_rewards
        ) = generate_uniform_trajectory(env)

        for k in range(kwargs['n_samples']):
            traj = {
                'context_actions': context_actions,
                'context_rewards': context_rewards,
                'optimal_action': env.opt_a,
                'price_grid': env.price_grid,
                'means': env.means,
                'alpha': env.alpha,
                'beta': env.beta,
            }
            trajs.append(traj)

    # Plot histogram of opt_a's
    opt_a_values = [np.argmax(traj['optimal_action']) for traj in trajs]
    plt.hist(opt_a_values, bins=10)
    plt.xlabel('opt_a')
    plt.ylabel('Frequency')
    plt.title('Histogram of opt_a')
    plt.savefig('opt_a_hist.png')

    return trajs



if __name__ == '__main__':
    np.random.seed(0)
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

    # Generate trajectories
    config = {
        'n_samples': n_samples,
        'horizon': horizon,
    }
    config.update({'dim': dim, 'var': var})
    train_trajs = generate_prices_histories(n_train_envs, **config)
    test_trajs = generate_prices_histories(n_test_envs, test=True, **config)
    eval_trajs = generate_prices_histories(n_eval_envs, test=True, **config)

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
