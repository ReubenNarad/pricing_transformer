import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time

import matplotlib.pyplot as plt
import torch

import numpy as np
import common_args
import random
import wandb
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.pytorch import loggers as pl_loggers

from dataset import Dataset
from net import Transformer
from utils import (
    build_prices_data_filename,
    build_prices_model_filename
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)

    parser.add_argument('--seed', type=int, default=0)

    args = vars(parser.parse_args())

    env = args['env']
    n_envs = args['envs']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    dataset_config = {
        'shuffle' : shuffle,
        'action_dim': action_dim,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'var': var,
        'cov': cov,
        'type': 'uniform',
        'store_gpu': True,
    }

    model_config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'lr': lr,
        'dim': dim,
        'seed': seed,
        'n_envs': n_envs,
        'test': False,
        'store_gpu': True,
    }

    path_train = build_prices_data_filename(
        env, n_envs, dataset_config, mode=0)
    path_test = build_prices_data_filename(
        env, n_envs, dataset_config, mode=1)
    filename = build_prices_model_filename(model_config)

    model = Transformer(model_config).to(device)

    params = {
        'batch_size': 64,
        'shuffle': shuffle,
    }

    logger = pl_loggers.WandbLogger(project='pricing_transformer')

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=[0],  # List of GPU IDs to use for training
        max_epochs=num_epochs,
        logger=logger,
        # callbacks=[pl.callbacks.ModelCheckpoint(dirpath='models/', filename=filename + "_epoch_" + str(num_epochs), save_top_k=1)]
    )

    log_filename = f'figs/loss/{filename}_logs.txt'
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)

    with open(log_filename, 'w') as f:
        pass

    train_dataset = Dataset(path_train, dataset_config)
    test_dataset = Dataset(path_test, dataset_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

