import torch
import torch.nn as nn
import transformers
from utils import build_prices_model_filename
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model

import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer(pl.LightningModule):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']
        self.lr = config['lr']

        self.filename = build_prices_model_filename(config)

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )

        #shape of action_seq: (batch_size, seq_len, action_dim)
        self.embed_transition = nn.Linear(
            self.action_dim + 1, self.n_embd)

        
        self.transformer = GPT2Model(config)
        
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        

    def forward(self, x):
        '''
        Forward pass of the pricing transformer network.
        
        Args:
            x (dict): Input data dictionary containing the following keys:
                - 'zeros' (torch.Tensor): Tensor of shape (batch_size, sequence_length, feature_dim) representing zeros.
                - 'context_actions' (torch.Tensor): Tensor of shape (batch_size, context_length, action_dim) representing context actions.
                - 'context_rewards' (torch.Tensor): Tensor of shape (batch_size, context_length, 1) representing context rewards.
        
        Returns:
            torch.Tensor: Predicted actions tensor of shape (batch_size, sequence_length - 1, action_dim) if self.test is False,
                          or tensor of shape (batch_size, 1, action_dim) if self.test is True.
        '''
        # See deploy_online_vec in evals/eval_prices.py
        # envs(=batch size) x horizon x actions (dim)
        zeros = x['zeros'][:, None, :]

        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [action_seq, reward_seq], dim=2)


        stacked_inputs = self.embed_transition(seq)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)

        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]


    def training_step(self, batch, batch_idx):
        true_actions = batch['optimal_actions']
        pred_actions = self.forward(batch)
        true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
        true_actions = true_actions.reshape(-1, self.action_dim)
        pred_actions = pred_actions.reshape(-1, self.action_dim)
        loss = self.loss_fn(pred_actions, true_actions)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        true_actions = batch['optimal_actions']
        pred_actions = self.forward(batch)
        true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
        true_actions = true_actions.reshape(-1, self.action_dim)
        pred_actions = pred_actions.reshape(-1, self.action_dim)
        loss = self.loss_fn(pred_actions, true_actions)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Calculate accuracy
        _, true_indices = true_actions.max(dim=1)
        _, pred_indices = pred_actions.max(dim=1)
        accuracy = (true_indices == pred_indices).float().mean()
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 100 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            torch.save(self.state_dict(), f'models/{self.filename}_epoch{self.current_epoch+1}.pt')