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

- `net.py`: Defines Transformer class, which utilizes a GPT-2 architecture. 

- `train.py`: Script for training Transformer models, includes comprehensive argument options, data loading, model training and evaluation, and generates related training loss plots.

- `utils.py`: Contains utilities for generating filenames for bandit data and model, initializing worker for data loading, and converting data to Tensor.


## Setup:

1. Install pytorch:
`pip install pytorch=1.13.0 cudatoolkit=11.7 -c pytorch -c nvidia`

2. Install dependencies:
`pip install -r requirements.txt`


## Example usage:

`
python3 collect_data.py --env prices --envs 1000 --H 200 --dim 20 --var 0.0 --lr 0.001 --layer 2 --head 2 --envs_eval 100 --seed 2 && \
python3 train.py --env prices --envs 1000 --H 200 --dim 20 --var 0.0 --lr 0.001 --layer 2 --head 2 --num_epochs 100 --seed 2 && \
python3 eval.py --env prices --envs 1000 --H 200  --dim 20 --var 0.0 --lr 0.001 --layer 2 --head 2 --epoch 100 --n_eval 100 --seed 2 
`
