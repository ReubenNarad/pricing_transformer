import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch
import numpy as np

import common_args
from evals import eval_prices
from net import Transformer
from utils import (
    build_prices_data_filename,
    build_prices_model_filename
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)
    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())

    n_envs = args['envs']
    n_samples = args['samples']
    H = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
    horizon = args['hor']
    n_eval = args['n_eval']
    seed = args['seed']
    lin_d = args['lin_d']
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)

    if test_cov < 0:
        test_cov = cov
    if horizon < 0:
        horizon = H

    config = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
        'lr': lr,
        'n_envs': n_envs,
        'dim': dim,
        'seed': seed,
    }

    # Load network from saved file.
    # By default, load the final file, otherwise load specified epoch.
    model = Transformer(config).to(device)
    filename = build_prices_model_filename(config)
    
    tmp_filename = filename
    if epoch < 0:
        model_path = f'models/{tmp_filename}.pt'
    else:
        model_path = f'models/{tmp_filename}_epoch{epoch}.pt'

    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_config = {
        'horizon': horizon,
        'dim': dim,
    }

    dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
    eval_filepath = build_prices_data_filename(
        envname, n_eval, dataset_config, mode=2)
    save_filename = f'{filename}_hor{horizon}.pkl'

    with open(eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)

    n_eval = min(n_eval, len(eval_trajs))


    evals_filename = f"evals_epoch{epoch}"
    if not os.path.exists(f'figs/{evals_filename}'):
        os.makedirs(f'figs/{evals_filename}', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/online'):
        os.makedirs(f'figs/{evals_filename}/online', exist_ok=True)

    # Online and offline evaluation.
    
    if envname == 'prices':
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
        }
        eval_prices.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        print(f"Saved figs/{evals_filename}/online/{save_filename}.png")
        plt.clf()
        plt.cla()
        plt.close()

        
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
        }

        with open(eval_filepath, 'rb') as f:
            eval_trajs = pickle.load(f)
