import matplotlib.pyplot as plt

import numpy as np
import scipy
import torch
from tqdm import tqdm
import pickle
import os

from ctrls.ctrl_prices import (
    BanditTransformerController,
    ParaThompsonSamplingPolicy,
)
from envs.prices_env import PricesEnv, PricesEnvVec

from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def deploy_online_vec(vec_env, controller, horizon, include_meta=False):
    num_envs = vec_env.num_envs
    # horizon x actions for each env since actions are one hot
    context_actions = np.zeros((num_envs, horizon, vec_env.du))
    # horizon x 1 for each env 
    context_rewards = np.zeros((num_envs, horizon, 1))
    envs = vec_env._envs

    cum_means = np.zeros((num_envs, horizon))
    print("Deplying online vectorized...")
    for h in range(horizon):
        batch = {
            'context_actions': context_actions[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
            'envs': envs
        }
        # converts batch to tensor, puts it in controller
        controller.set_batch_numpy_vec(batch)
        #gets result at time h
        actions_lnr, rewards_lnr = vec_env.deploy(controller)

        context_actions[:, h, :] = actions_lnr
        context_rewards[:, h, :] = rewards_lnr[:,None]

        action_indices = np.argmax(actions_lnr, axis=1)
        revenues = rewards_lnr * np.array([env.price_grid[a] for env, a in zip(envs, action_indices)])
        cum_means[:, h] = revenues

    print("Depolyed online vectorized")
    
    if not include_meta:
        return cum_means
    else:
        meta = {
            'context_actions': context_actions,
            'context_rewards': context_rewards,
        }
        return cum_means, meta


def online(eval_trajs, model, n_eval, horizon, var):
    print("Starting Online ...")

    all_means = {}
    metas = {}

    envs = []
    print("Creating envs ...")
    for i_eval in tqdm(range(n_eval)):
        traj = eval_trajs[i_eval]
        # Note traj['price_grid'] is a list of prices
        env = PricesEnv(traj['alpha'], traj['beta'], len(traj['price_grid']), 
                        horizon, var=var, lower_price=5, upper_price=10)
        envs.append(env)
    vec_env = PricesEnvVec(envs)
    
    # controller = OptPolicy(envs, batch_size=len(envs))
    # print("Deploying online opt ...")
    # cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
    # assert cum_means.shape[0] == n_eval
    all_means['opt'] = np.array([[env.opt_r]*horizon for env in envs])
    #all_means['opt'] = cum_means
    metas['opt'] = [[env.opt_r]*horizon for env in envs]

    controller = BanditTransformerController(
        model,
        sample=True,
        batch_size=len(envs))
    print("Deploying online transformer ...")
    cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
    assert cum_means.shape[0] == n_eval
    all_means['Transformer'] = cum_means
    metas['Transformer'] = meta

    # controller = ParaThompsonSamplingPolicy(
    #     envs[0],
    #     std=var,
    #     theta_0=[0, 0],
    #     cov_0=np.eye(2),
    #     warm_start=False,
    #     batch_size=len(envs))
    # print("Deploying online Thompson ...")
    # cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
    # assert cum_means.shape[0] == n_eval
    # all_means['ParamThomp'] = cum_means
    # metas['ParamThomp'] = meta

    # controller = UCBPolicy(
    #     envs[0],
    #     const=1.0,
    #     batch_size=len(envs))
    # cum_means = deploy_online_vec(vec_env, controller, horizon).T
    # assert cum_means.shape[0] == n_eval
    # all_means['UCB'] = cum_means
    # metas['UCB'] = meta

    # controller = LinUCBPolicy(
    #     envs[0],
    #     const=1.0,
    #     batch_size=len(envs)
    # )
    # cum_means = deploy_online_vec(vec_env, controller, horizon).T
    # assert cum_means.shape[0] == n_eval
    # all_means['LinearUCB'] = cum_means
    # metas['LinearUCB'] = meta

    # Pickle meta
    
    if not os.path.exists('metas'):
        os.makedirs('metas')

    with open(f'metas/H_{horizon}_dim_{envs[0].dim}_meta.pkl', 'wb') as f:
        pickle.dump(metas, f)  # Fixed the closing parenthesis

    all_means = {k: np.array(v) for k, v in all_means.items()}
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}

    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}

    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


    for key in means.keys():
        if key == 'opt':
            ax1.plot(means[key], label=key, linestyle='--',
                    color='black', linewidth=2)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2, color='black')
        else:
            ax1.plot(means[key], label=key)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], means[key] + sems[key], alpha=0.2)


    ax1.set_yscale('log')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Simple Regret')
    ax1.set_title(f'Online Evaluation, mean of {n_eval} trajectories')
    ax1.legend()


    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key)
            ax2.fill_between(np.arange(horizon), regret_means[key] - regret_sems[key], regret_means[key] + regret_sems[key], alpha=0.2)

    # ax2.set_yscale('log')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title(f'Cumuative regret, H={horizon}')
    ax2.legend()
