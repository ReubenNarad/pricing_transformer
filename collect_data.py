import os
import pickle

import numpy as np
import argparse
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import common_args
from envs import prices_env
from utils import build_prices_data_filename, build_run_name
from net import Transformer
from ctrls.ctrl_prices import Controller, ParaThompsonSamplingPolicy, TransformerController
from dataset import Dataset


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

def generate_thmp_trajectory(env, controller=None):
    '''
    Create a trajectory of a controller policy.

    Input:
    env: PricesEnv object
    controller: Controller object

    Outputs:
    us: H x env.dim actions as one hot encoded vectors
    rs: H x 1 rewards
    '''

    controller = ParaThompsonSamplingPolicy(env)

    # Initialize the controller with the environment
    controller.envs = [env]
    controller.reset()

    # For logging trajectories
    us, rs = [], []

    H = env.H_context
    for t in range(H):
        # Get the action from the controller
        actions, _ = controller.act_numpy_vec()
        action = actions[0]  # Since we are dealing with a single trajectory, take the first batch element

        # Step the environment with the chosen action
        reward, _, _ = env.step(action)

        # Log the action and reward
        us.append(action)
        rs.append(reward)

        # Update the controller's posterior with the observed action and reward
        controller.update_posterior(0, [action], [reward])


    us, rs = np.array(us), np.array(rs)
    return us, rs


def generate_transformer_trajectory(env, controller):
    '''
    Create a trajectory from a transformer policy.

    Input:
    env: PricesEnv object
    controller: Controller object

    Outputs:
    us: H x env.dim actions as one hot encoded vectors
    rs: H x 1 rewards
    '''
    
    # Initialize the controller with the environment
    controller.envs = [env]

    us, rs = [], []
    H = env.H_context
   
    for t in range(H):
        # Get the action from the controller
        batch = {
            'context_actions': np.expand_dims(np.array(us), axis=0),
            'context_rewards': np.expand_dims(np.array(rs), axis=0),
        }
       
        if len(us) > 0:
            batch['context_actions'] = np.expand_dims(batch['context_actions'], axis=-1)
            batch['context_rewards'] = np.expand_dims(batch['context_rewards'], axis=-1)
 
        controller.set_batch_numpy_vec(batch)
       
        actions, _ = controller.act_numpy_vec()
       
        reward, _, _ = env.step(actions)
       
        us.append(actions)
        rs.append(reward)
   
    us, rs = np.array(us), np.array(rs)
   
    return us, rs


def generate_prices_histories(n_envs, dim, horizon, var, env_type='prices', test=False, transformer=False, **kwargs):
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
   
    if transformer:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = 'runs/d20_envs50000_H200_var0.0_head2_layer2_embd32_lr0.001_seed2/model_epoch100.pt'
        model_config = {
            'H': 200,
            'state_dim': 1,
            'action_dim': 20,
            'layer': 2,
            'embd': 32,
            'head': 2,
            'var': 0.0,
            'test': True,
            'lr': 0.001,
            'envs': 50000,
            'dim': 20,
            'seed': 2,
            'dropout': False,
        }
        model = Transformer(model_config).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        ctrl = TransformerController(model, batch_size=1)


    trajs = []
    for env in tqdm(envs):
        (
            context_actions,
            context_rewards
        ) = generate_uniform_trajectory(env)
        # ) = generate_thmp_trajectory(env)
        # ) = generate_transformer_trajectory(env, ctrl)

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
    
    # visualize actions of laset trajectory as 1-0 heatmap
    last_traj = trajs[-1]
    opt_a = last_traj['optimal_action']
    actions = last_traj['context_actions']

    plt.figure(figsize=(10, 6))
    sns.heatmap(actions, cmap='viridis', cbar=False)
    plt.xlabel('Price Grid')
    plt.ylabel('Time Step')
    plt.title(f'Optimal Action: {np.argmax(opt_a)}')
    plt.savefig('data_traj_example.png')
    plt.show()

    return trajs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_run_args(parser)
    args = vars(parser.parse_args())
    print("Dataset Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    lr = args['lr']
    layer = args['layer']
    head = args['head']
    embd = args['embd']
    seed = args['seed']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']
    lin_d = args['lin_d']

    np.random.seed(seed)
    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    # Generate trajectories
    data_config = {
        'n_samples': n_samples,
        'horizon': horizon,
    }
    data_config.update({'dim': dim, 'var': var})

    train_trajs = generate_prices_histories(n_train_envs, test=False, **data_config)
    test_trajs = generate_prices_histories(n_test_envs, test=True, **data_config)
    eval_trajs = generate_prices_histories(n_eval_envs, test=True, **data_config)

    run_filepath = build_run_name(args)

    train_filepath = build_prices_data_filename(n_envs, data_config, mode=0)
    test_filepath = build_prices_data_filename(n_envs, data_config, mode=1)
    eval_filepath = build_prices_data_filename( n_eval_envs, data_config, mode=2)

    # Save to run/datasets
    path = f'runs/{run_filepath}'
    if not os.path.exists(f'runs/{run_filepath}'):
        os.makedirs(f'runs/{run_filepath}', exist_ok=True)
    with open(path+'/'+train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(path+'/'+test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(path+'/'+eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to runs/{run_filepath}/{train_filepath}.")
    print(f"Saved to runs/{run_filepath}/{test_filepath}.")
    print(f"Saved to runs/{run_filepath}/{eval_filepath}.")
