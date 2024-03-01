import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import numpy as np
import common_args
import random
import wandb
from dataset import Dataset, ImageDataset
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    worker_init_fn,
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
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
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
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }
    
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="pricing_transformer",

#         # track hyperparameters and run metadata
#         config={
#         "learning_rate": lr,
#         "architecture": "Transformer",
#         "dataset": "Custom",
#         "epochs": num_epochs,
#         }
#     )
    
    if env == 'prices':
        state_dim = 1
        
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    
    elif env == 'bandit':
        state_dim = 1
        

        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'bandit_thompson':
        state_dim = 1

        dataset_config.update({'var': var, 'cov': cov, 'type': 'bernoulli'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'linear_bandit':
        state_dim = 1

        dataset_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        path_train = build_linear_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_linear_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(env, model_config)

    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
    }
    
    model = Transformer(config).to(device)

    params = {
        'batch_size': 64,
        'shuffle': True,
    }

    log_filename = f'figs/loss/{filename}_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)

    train_dataset = Dataset(path_train, config)
    test_dataset = Dataset(path_test, config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    test_loss = []
    train_loss = []

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))

    for epoch in range(num_epochs):
        # EVALUATION
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['optimal_actions']
                pred_actions = model(batch)
                true_actions = true_actions.unsqueeze(
                    1).repeat(1, pred_actions.shape[1], 1)
                true_actions = true_actions.reshape(-1, action_dim)
                pred_actions = pred_actions.reshape(-1, action_dim)

                loss = loss_fn(pred_actions, true_actions)
                epoch_test_loss += loss.item() / horizon

        test_loss.append(epoch_test_loss / len(test_dataset))
        end_time = time.time()
        printw(f"\tTest loss: {test_loss[-1]}")
        printw(f"\tEval time: {end_time - start_time}")
        
        # wandb.log({"Test Loss": test_loss[-1]})



        # TRAINING
        epoch_train_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # print(f"CONTEXT STATES: {batch['context_states']}, \n CONTEXT ACTIONS: {batch['context_actions']}")
            
            true_actions = batch['optimal_actions']
            pred_actions = model(batch)
            
            # print(f"PRED: {pred_actions} \n TRUE: {true_actions}")
            
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_actions.shape[1], 1)
            true_actions = true_actions.reshape(-1, action_dim)
            pred_actions = pred_actions.reshape(-1, action_dim)

            optimizer.zero_grad()
            loss = loss_fn(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon

        train_loss.append(epoch_train_loss / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain time: {end_time - start_time}")


        # LOGGING
        if (epoch + 1) % 50 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
            torch.save(model.state_dict(),
                       f'models/{filename}_epoch{epoch+1}.pt')

        # PLOTTING
        if (epoch + 1) % 10 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Test Loss:        {test_loss[-1]}")
            printw(f"Train Loss:       {train_loss[-1]}")
            printw("\n")

            plt.yscale('log')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.plot(test_loss[1:], label="Test Loss")
            plt.legend()
            plt.savefig(f"figs/loss/{filename}_train_loss.png")
            plt.clf()

    torch.save(model.state_dict(), f'models/{filename}.pt')
    print("Done.")
    # wandb.finish()
