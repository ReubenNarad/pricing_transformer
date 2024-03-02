import argparse
import os
import pickle
import random

import gym
import numpy as np
from skimage.transform import resize
from IPython import embed

import common_args
from envs import bandit_env, prices_env
from ctrls.ctrl_bandit import ThompsonSamplingPolicy
from evals import eval_bandit
from utils import build_bandit_data_filename

from tqdm import tqdm
                  

def rollin_bandit(env, cov, orig=False):
    H = env.H_context
    opt_a_index = env.opt_a_index
    xs, us, xps, rs = [], [], [], []
    cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    alpha = np.ones(env.dim)
    probs = np.random.dirichlet(alpha)
    probs2 = np.zeros(env.dim)
    rand_index = np.random.choice(np.arange(env.dim))
    probs2[rand_index] = 1.0
    probs = (1 - cov) * probs + cov * probs2


    for h in range(H):
        x = np.array([1])
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim), p=probs)
        u[i] = 1.0
        xp, r = env.transit(x, u)

        xs.append(x)
        us.append(u)
        xps.append(xp)
        rs.append(r)

    xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
    return xs, us, xps, rs


def generate_bandit_histories_from_envs(envs, n_hists, n_samples, cov, type):
    trajs = []
    for env in tqdm(envs):
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin_bandit(env, cov=cov)
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'means': env.means,
                }
                trajs.append(traj)
    return trajs


def generate_bandit_histories(n_envs, dim, horizon, var, **kwargs):
    envs = [bandit_env.sample(dim, horizon, var)
            for _ in range(n_envs)]
    trajs = generate_bandit_histories_from_envs(envs, **kwargs)
    return trajs


def generate_linear_bandit_histories(n_envs, dim, lin_d, horizon, var, **kwargs):
    # generate fixed features for arms of all linear bandits
    rng = np.random.RandomState(seed=1234)
    arms = rng.normal(size=(dim, lin_d)) / np.sqrt(lin_d)

    envs = [bandit_env.sample_linear(arms, horizon, var)
            for _ in range(n_envs)]

    data_type = kwargs['data_type']

    print("Generating histories...")
    if data_type=='thompson':
        context_states_all, context_actions_all, context_next_states_all, context_rewards_all = [], [], [], []
        for j in range(n_hists):
            context_states, context_actions, context_next_states, context_rewards = rollin_linear_bandit_vec(envs)

            context_states_all.append(context_states)
            context_actions_all.append(context_actions)
            context_next_states_all.append(context_next_states)
            context_rewards_all.append(context_rewards)
        context_states_all = np.stack(context_states_all, axis=1)
        context_actions_all = np.stack(context_actions_all, axis=1)
        context_next_states_all = np.stack(context_next_states_all, axis=1)
        context_rewards_all = np.stack(context_rewards_all, axis=1)


    trajs = []
    for i, env in enumerate(envs):
        print('Generating linear bandit histories for env {}/{}'.format(i+1, n_envs))
        for j in range(n_hists):
            if data_type=='uniform':
                context_states, context_actions, context_next_states, context_rewards = rollin_bandit(env, cov=0.0)
            elif data_type=='thompson':
                context_states = context_states_all[i, j]
                context_actions = context_actions_all[i, j]
                context_next_states = context_next_states_all[i, j]
                context_rewards = context_rewards_all[i, j]
            else:
                raise ValueError("Invalid data type")

            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'means': env.means,
                    'arms': arms,
                    'theta': env.theta,
                    'var': env.var,
                }
                trajs.append(traj)
    return trajs


###########################################################################################
# PRICES
###########################################################################################

def rollin_prices(env, cov, orig=False):
    H = env.H_context
    opt_a_index = env.opt_a_index
    xs, us, xps, rs = [], [], [], []
    cov = np.random.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    alpha = np.ones(env.dim)

    for h in range(H):
        x = np.array([1])
        u = np.zeros(env.dim)
        i = np.random.choice(np.arange(env.dim))
        
        u[i] = 1.0
        xp, r = env.transit(x, u)

        xs.append(x)
        us.append(u)
        xps.append(xp)
        rs.append(r)

    xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
    return xs, us, xps, rs

def generate_prices_histories_from_envs(envs, n_hists, n_samples, cov, type):
    trajs = []
    for env in tqdm(envs):
        for j in range(n_hists):
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin_prices(env, cov=cov)
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a

                traj = {
                    'query_state': query_state,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'optimal_action': optimal_action,
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
    trajs = generate_prices_histories_from_envs(envs, **kwargs)
    return trajs

###########################################################################################

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

    ###########    ###########    ###########    ###########    ###########    ###########
    
    if env == 'prices':
        config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})

        train_trajs = generate_prices_histories(n_train_envs, **config)
        test_trajs = generate_prices_histories(n_test_envs, **config)
        eval_trajs = generate_prices_histories(n_eval_envs, **config)
                        
        train_filepath = build_bandit_data_filename(env, n_envs, config, mode=0)
        test_filepath = build_bandit_data_filename(env, n_envs, config, mode=1)
        eval_filepath = build_bandit_data_filename(env, n_eval_envs, config, mode=2)
   
    ###########    ###########    ###########    ###########    ###########    ###########

    elif env == 'bandit':
        config.update({'dim': dim, 'var': var, 'cov': cov, 'type': 'uniform'})

        train_trajs = generate_bandit_histories(n_train_envs, **config)
        test_trajs = generate_bandit_histories(n_test_envs, **config)
        eval_trajs = generate_bandit_histories(n_eval_envs, **config)
                
        train_filepath = build_bandit_data_filename(env, n_envs, config, mode=0)
        test_filepath = build_bandit_data_filename(env, n_envs, config, mode=1)
        eval_filepath = build_bandit_data_filename(env, n_eval_envs, config, mode=2)

    elif env == 'linear_bandit':
        config.update({'dim': dim, 'lin_d': lin_d, 'var': var, 'cov': cov, 'data_type': 'thompson'})

        train_trajs = generate_linear_bandit_histories(n_train_envs, **config)
        test_trajs = generate_linear_bandit_histories(n_test_envs, **config)
        eval_trajs = generate_linear_bandit_histories(n_eval_envs, **config)

        train_filepath = build_linear_bandit_data_filename(env, n_envs, config, mode=0)
        test_filepath = build_linear_bandit_data_filename(env, n_envs, config, mode=1)
        eval_filepath = build_linear_bandit_data_filename(env, n_eval_envs, config, mode=2)
        
    else:
        raise NotImplementedError


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
