import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    torch.manual_seed(worker_seed)
    numpy_seed = int(worker_seed % (2**32 - 1))  # Optional, in case you also use numpy in the DataLoader
    np.random.seed(numpy_seed)


def build_run_name(config):
    """
    Builds the filename for the run.
    """
    filename = ''
    if 'dim' in config:
        filename += 'd' + str(config['dim'])
    if 'envs' in config:
        filename += '_envs' + str(config['envs'])
    if 'H' in config:
        filename += '_H' + str(config['H'])
    if 'var' in config:
        filename += '_var' + str(config['var'])
    if 'head' in config:
        filename += '_head' + str(config['head'])
    if 'layer' in config:
        filename += '_layer' + str(config['layer'])
    if 'embd' in config:
        filename += '_embd' + str(config['embd'])
    if 'lr' in config:
        filename += '_lr' + str(config['lr'])
    if 'seed' in config:
        filename += '_seed' + str(config['seed'])
    return filename

def build_prices_data_filename(n_envs, config, mode):
    """
    Builds the filename for the prices data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename = 'trajs'
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_eval'
    return f"{filename}.pkl"


def build_prices_model_filename(config):
    """
    Builds the filename for model.
    """

    filename = 'model'
    filename += '_d' + str(config['dim'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_H' + str(config['horizon'])
    filename += '_head' + str(config['n_head'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_lr' + str(config['lr'])
    filename += '_seed' + str(config['seed'])
    return filename



def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()
    


