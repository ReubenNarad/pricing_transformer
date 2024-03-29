import matplotlib.pyplot as plt

import numpy as np
import scipy
import torch
from tqdm import tqdm

from ctrls.ctrl_prices import (
    BanditTransformerController,
    GreedyOptPolicy,
    EmpMeanPolicy,
    OptPolicy,
    PessMeanPolicy,
    ParaThompsonSamplingPolicy,
    UCBPolicy,
    LinUCBPolicy
)
from envs.prices_env import PricesEnv, PricesEnvVec

from utils import convert_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deploy_online(env, controller, horizon):    
    context_actions = torch.zeros((1, horizon, env.du)).float().to(device)
    context_rewards = torch.zeros((1, horizon, 1)).float().to(device)

    cum_means = []
    for h in range(horizon):
        batch = {
            'context_actions': context_actions[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
        }

        controller.set_batch(batch)
        actions_lnr, rewards_lnr = env.deploy(
            controller)

        context_actions[0, h, :] = convert_to_tensor(actions_lnr[0])
        context_rewards[0, h, :] = convert_to_tensor(rewards_lnr[0])

        actions = actions_lnr.flatten()
        mean = env.get_arm_value(actions)

        cum_means.append(mean)

    return np.array(cum_means)


def deploy_online_vec(vec_env, controller, horizon, include_meta=False):
    num_envs = vec_env.num_envs
    context_actions = np.zeros((num_envs, horizon, vec_env.du))
    context_rewards = np.zeros((num_envs, horizon, 1))
    envs = vec_env._envs

    cum_means = []
    print("Deplying online vectorized...")
    for h in range(horizon):
        batch = {
            'context_actions': context_actions[:, :h, :],
            'context_rewards': context_rewards[:, :h, :],
            'envs': envs
        }
        controller.set_batch_numpy_vec(batch)
        actions_lnr, rewards_lnr = vec_env.deploy(controller)

        context_actions[:, h, :] = actions_lnr
        context_rewards[:, h, :] = rewards_lnr[:,None]

        action_indices = np.argmax(actions_lnr, axis=1)
        revenues = rewards_lnr * np.array([env.price_grid[a] for env, a in zip(envs, action_indices)])

        cum_means.append(revenues)

    print("Depolyed online vectorized")
    
    cum_means = np.array(cum_means)
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
        means = traj['means']
        
        env = PricesEnv(traj['alpha'], traj['beta'], len(traj['prices']), horizon, var=var, lower_price=5, upper_price=10)
        envs.append(env)

    vec_env = PricesEnvVec(envs)
    
    controller = OptPolicy(
        envs,
        batch_size=len(envs))
    print("Deploying online opt ...")
    cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
    cum_means = cum_means.T
    assert cum_means.shape[0] == n_eval
    all_means['opt'] = cum_means
    metas['opt'] = meta

    controller = BanditTransformerController(
        model,
        sample=True,
        batch_size=len(envs))
    print("Deploying online transformer ...")
    cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
    cum_means = cum_means.T
    assert cum_means.shape[0] == n_eval
    all_means['Transformer'] = cum_means
    metas['Transformer'] = meta

    controller = ParaThompsonSamplingPolicy(
        envs[0],
        std=var,
        theta_0=[5, -1.5],
        cov_0=np.eye(2),
        warm_start=False,
        batch_size=len(envs))
    print("Deploying online Thompson ...")
    cum_means, meta = deploy_online_vec(vec_env, controller, horizon, include_meta=True)
    cum_means = cum_means.T
    assert cum_means.shape[0] == n_eval
    all_means['ParamThomp'] = cum_means
    metas['ParamThomp'] = meta

    controller = UCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['UCB'] = cum_means
    metas['UCB'] = meta

    controller = LinUCBPolicy(
        envs[0],
        const=1.0,
        batch_size=len(envs)
    )
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means['LinearUCB'] = cum_means
    metas['LinearUCB'] = meta

    # Pickle meta
    import pickle
    import os

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

def offline(eval_trajs, model, n_eval, horizon, var):
    all_rs_lnr = []
    all_rs_greedy = []
    all_rs_opt = []
    all_rs_emp = []
    all_rs_pess = []
    all_rs_thmp = []

    num_envs = len(eval_trajs)

    tmp_env = PricesEnv(eval_trajs[0]['alpha'], eval_trajs[0]['beta'], len(eval_trajs[0]['prices']), horizon, var=var)
    context_actions = np.zeros((num_envs, horizon, tmp_env.du))
    context_rewards = np.zeros((num_envs, horizon, 1))

    envs = []

    for i_eval in range(n_eval):
        # print(f"Eval traj: {i_eval}")
        traj = eval_trajs[i_eval]
        means = traj['means']

        env = PricesEnv(traj['alpha'], traj['beta'], len(traj['prices']), horizon, var=var)
        envs.append(env)

        context_actions[i_eval, :, :] = traj['context_actions'][:horizon]
        context_rewards[i_eval, :, :] = traj['context_rewards'][:horizon,None]


    vec_env = PricesEnvVec(envs)
    batch = {
        'context_actions': context_actions,
        'context_rewards': context_rewards,
    }

    opt_policy = OptPolicy(envs, batch_size=num_envs)
    # emp_policy = EmpMeanPolicy(envs[0], online=False, batch_size=num_envs)
    lnr_policy = BanditTransformerController(model, sample=False, batch_size=num_envs)
    thomp_policy = ParaThompsonSamplingPolicy(
        envs[0],
        std=var,
        sample=False,
        prior_mean=0.5,
        prior_var=1/12.0,
        warm_start=False,
        batch_size=num_envs)
    lcb_policy = PessMeanPolicy(
        envs[0],
        const=.8,
        batch_size=len(envs))


    opt_policy.set_batch_numpy_vec(batch)
    # emp_policy.set_batch_numpy_vec(batch)
    thomp_policy.set_batch_numpy_vec(batch)
    lcb_policy.set_batch_numpy_vec(batch)
    lnr_policy.set_batch_numpy_vec(batch)
    
    _, rs_opt = vec_env.deploy_eval(opt_policy)
    # _, _, _, rs_emp = vec_env.deploy_eval(emp_policy)
    _, rs_lnr = vec_env.deploy_eval(lnr_policy)
    _, rs_lcb = vec_env.deploy_eval(lcb_policy)
    _, rs_thmp = vec_env.deploy_eval(thomp_policy)


    baselines = {
        'opt': np.array(rs_opt),
        'transformer': np.array(rs_lnr),
        # 'greedy': np.array(rs_emp),
        'Parameterized TS': np.array(rs_thmp),
        'lcb': np.array(rs_lcb),
    }    
    baselines_means = {k: np.mean(v) for k, v in baselines.items()}
    colors = plt.cm.viridis(np.linspace(0, 1, len(baselines_means)))
    plt.bar(baselines_means.keys(), baselines_means.values(), color=colors)
    plt.title(f'Mean Reward on {n_eval} Trajectories')


    return baselines


def offline_graph(eval_trajs, model, n_eval, horizon, var):
    horizons = np.linspace(1, horizon, 50, dtype=int)

    all_means = []
    all_sems = []
    
    print("Starting offline ...")
    for h in tqdm(horizons):
        config = {
            'horizon': h,
            'var': var,
            'n_eval': n_eval,
        }
        config['horizon'] = h
        baselines = offline(eval_trajs, model, **config)
        plt.clf()

        means = {k: np.mean(v, axis=0) for k, v in baselines.items()}
        sems = {k: scipy.stats.sem(v, axis=0) for k, v in baselines.items()}
        all_means.append(means)


    for key in means.keys():
        if not key == 'opt':
            regrets = [all_means[i]['opt'] - all_means[i][key] for i in range(len(horizons))]            
            plt.plot(horizons, regrets, label=key)
            plt.fill_between(horizons, regrets - sems[key], regrets + sems[key], alpha=0.2)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Dataset size')
    plt.ylabel('Simple Regret')
    config['horizon'] = horizon
