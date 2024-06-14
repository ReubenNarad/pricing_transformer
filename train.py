import torch
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
import numpy as np
import common_args
import random
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.pytorch import loggers as pl_loggers

from dataset import Dataset
from net import Transformer
from utils import (
    build_prices_data_filename,
    build_prices_model_filename,
    build_run_name
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    max_length = max(len(item['context_actions']) for item in batch)
    
    padded_batch = {
        'context_actions': [],
        'context_rewards': [],
        'optimal_actions': [],
        'zeros': [],
    }
    
    for item in batch:
        context_actions = item['context_actions']
        context_rewards = item['context_rewards']
        optimal_actions = item['optimal_actions']
        zeros = item['zeros']
        
        # Pad sequences
        pad_length = max_length - len(context_actions)
        action_pad = torch.zeros(pad_length, context_actions.shape[1]).to(device)
        reward_pad = torch.zeros(pad_length, context_rewards.shape[1]).to(device)
        
        padded_context_actions = torch.cat([context_actions, action_pad], dim=0)
        padded_context_rewards = torch.cat([context_rewards, reward_pad], dim=0)
        
        padded_batch['context_actions'].append(padded_context_actions)
        padded_batch['context_rewards'].append(padded_context_rewards)
        padded_batch['optimal_actions'].append(optimal_actions)
        padded_batch['zeros'].append(zeros)
    
    # Stack tensors
    for key in padded_batch:
        padded_batch[key] = torch.stack(padded_batch[key])
    
    return padded_batch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    common_args.add_run_args(parser)

    common_args.add_train_args(parser)

    args = vars(parser.parse_args())

    env = args['env']
    envs = args['envs']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    embd = args['embd']
    head = args['head']
    layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    
    # Set seeds
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

    run_config = {
        'shuffle' : shuffle,
        'n_samples': n_samples,
        'H': horizon,
        'dim': dim,
        'var': var,
        'cov': cov,
        'type': 'uniform',
        'store_gpu': True,
        'truncate': True,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'layer': layer,
        'embd': embd,
        'head': head,
        'dropout': dropout,
        'lr': lr,
        'seed': seed,
        'envs': envs,
        'test': False,
    }

    # dataset_config = {
    #     'shuffle' : shuffle,
    #     'action_dim': action_dim,
    #     'n_samples': n_samples,
    #     'horizon': horizon,
    #     'dim': dim,
    #     'var': var,
    #     'cov': cov,
    #     'type': 'uniform',
    #     'store_gpu': True,
    #     'truncate': True
    # }

    # model_config = {
    #     'horizon': horizon,
    #     'state_dim': state_dim,
    #     'action_dim': action_dim,
    #     'n_layer': n_layer,
    #     'n_embd': n_embd,
    #     'n_head': n_head,
    #     'shuffle': shuffle,
    #     'dropout': dropout,
    #     'lr': lr,
    #     'dim': dim,
    #     'seed': seed,
    #     'n_envs': n_envs,
    #     'test': False,
    #     'store_gpu': True,
    # }

    train_filename = build_prices_data_filename(
        envs, run_config, mode=0)
    test_filename = build_prices_data_filename(
        envs, run_config, mode=1)
    
    run_name = build_run_name(run_config)

    model = Transformer(run_config).to(device)

    params = {
        'batch_size': 64,
        'shuffle': shuffle,
    }

    # Build wandb run name
    wandb_run_name = [
    run_name,
    time.strftime('%Y%m%d-%H%M%S')
    ]

    wandb_run_name = " ".join(wandb_run_name)

    logger = pl_loggers.WandbLogger(project='pricing_transformer', name=wandb_run_name)

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=[0],  # List of GPU IDs to use for training
        max_epochs=num_epochs,
        logger=logger,
        enable_checkpointing=False
    )


    # Load the dataset
    train_dataset = Dataset(f'runs/{run_name}/'+train_filename, run_config)
    test_dataset = Dataset(f'runs/{run_name}/'+test_filename, run_config)

    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, **params)

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

