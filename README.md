# pricing_transformer
Research project, implementing Lee et. al.'s (https://arxiv.org/pdf/2306.14892.pdf) Decision Pretrained Transformer (DPT) for dynamic pricing

## Notes:

- The base transformer model is GPT2 via 🤗 Transformers
- We use wandb for tracking loss on runs

## File explanations:

- `/evals`: Contains scripts for evaluating models under different environments

- `/envs`: Contains scripts defining environments

- `/ctrls`: Contains scripts for controlling agents during evaluation

- `collect_data.py`: Script for generating environment trajectories and storing them as datasets.

- `common_args.py`: Parses command-line arguments related to dataset configuration, model architecture, training settings, and evaluation parameters.

- `dataset.py`: Implements the Dataset class for loading, transforming, and retrieving trajectory samples in both non-image-based and image-based (for miniworld) formats for RL experiments.

- `eval.py`: Script for evaluating model and other policies, and plotting results

- `net.py`: Defines neural network with Transformer and ImageTransformer (for miniworld) classes, both of which utilize a GPT-2 model for processing trajectory sequences.

- `train.py`: Script for training Transformer models, includes comprehensive argument options, data loading, model training and evaluation, and generates related training loss plots.

- `utils.py`: Contains utilities for generating filenames for bandit data and model, initializing worker for data loading, and converting data to Tensor.


## Setting up environment:

1. Install pytorch:
`pip install pytorch=1.13.0 torchvision=0.14.0 cudatoolkit=11.7 -c pytorch -c nvidia`

2. Install dependencies:
`pip install -r requirements.txt`


## Example usage:

1. Collect data:
`python3 collect_data.py --env prices --envs 1000 --H 50 --dim 10 --var 0.3 --cov 0.0 --envs_eval 200`

2. Train:
`python3 train.py --env prices --envs 1000 --H 50 --dim 10 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --num_epochs 100 --seed 1`

3. Evaluate:
`python3 eval.py --env prices --envs 1000 --H 50 --dim 10 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch 100 --n_eval 1000 --seed 1`

`
python3 collect_data.py --env prices --envs 10000 --H 100 --dim 10 --var 0.01 --envs_eval 200 && \
python3 train.py --env prices --envs 10000 --H 100 --dim 10 --var 0.01 --lr 0.0001 --layer 4 --head 4 --shuffle --num_epochs 500 --seed 1 && \
python3 eval.py --env prices --envs 10000 --H 100  --dim 10 --var 0.01 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch 500 --n_eval 200 --seed 1
`

# Swtiching to multi-product todos
x Collect data needs to take --env arguments again (for env multi_prices)
x Create multi_prices_env.py
o Create eval_multi_prices
o Create baseline controller (param thomp, LinUCB)

