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
    build_prices_model_filename,
    build_run_name
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_run_args(parser)
    common_args.add_eval_args(parser)

    args = vars(parser.parse_args())

    envs = args['envs']
    n_samples = args['samples']
    H = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    embd = args['embd']
    head = args['head']
    layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
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

    run_config = {
        'H': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'layer': layer,
        'embd': embd,
        'head': head,
        'var': var,
        'dropout': dropout,
        'test': True,
        'lr': lr,
        'envs': envs,
        'dim': dim,
        'seed': seed,
    }

    # Load network from saved file.
    # By default, load the final file, otherwise load specified epoch.
    model = Transformer(run_config).to(device)
    filename = 'model'

    run_name = build_run_name(run_config)
    
    if epoch < 0:
        model_path = f'runs/{run_name}/model.pt'
    else:
        model_path = f'runs/{run_name}/model_epoch{epoch}.pt'

    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    run_name = build_run_name(run_config)

    eval_filepath = build_prices_data_filename(
        n_eval, run_config, mode=2)
    save_filename = f'{filename}_hor{H}.pkl'

    with open(f'runs/{run_name}/'+eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)

    n_eval = min(n_eval, len(eval_trajs))

    if not os.path.exists(f'runs/{run_name}/evals'):
        os.makedirs(f'runs/{run_name}/evals', exist_ok=True)

    # Online and offline evaluation.
    
    if envname == 'prices':
        eval_config = {
            'horizon': H,
            'var': var,
            'n_eval': n_eval,
            'run_name': run_name,
        }
        eval_prices.online(eval_trajs, model, **eval_config)
        plt.savefig(f'runs/{run_name}/evals/online_epoch{epoch}.png')
        print(f"Saved runs/{run_name}/evals/online_epoch{epoch}.png")
        plt.clf()
        plt.cla()
        plt.close()
