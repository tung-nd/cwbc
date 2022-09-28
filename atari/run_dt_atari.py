import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.rvs_atari import RVS
from mingpt.trainer_rvs_atari import RVSTrainer
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--lr', type=float, default=6e-4)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')

# conservative 
parser.add_argument('--conservative_percentile', type=float, default=None)
parser.add_argument('--conservative_std', type=float, default=50.0)
parser.add_argument('--conservative_w', type=float, default=1.0)

args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, returns):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.returns = returns
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for traj_id, i in enumerate(self.done_idxs):
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps, self.returns[traj_id]

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps, returns)

epochs = args.epochs

if args.model_type == 'rvs':
    model = RVS(n_embed=128, act_dim=train_dataset.vocab_size, hidden_size=1024, n_layer=4, dropout=0.1)
    sorted_returns = np.sort(returns)
    
    if args.conservative_percentile is not None:
        conservative_level = math.sqrt(12*(args.conservative_std**2))
        conservative_return = sorted_returns[int(args.conservative_percentile * len(sorted_returns))]
        conservative_w = args.conservative_w
        print ('conservative return', conservative_return)
    else:
        conservative_level = None
        conservative_return = None
        conservative_w = None

    trainer = RVSTrainer(
        model, args.game, train_dataset, epochs, args.batch_size, lr=args.lr, decay=1e-4,
        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
        conservative_return=conservative_return, conservative_level=conservative_level, conservative_w=conservative_w,
        max_return=sorted_returns[-1], num_workers=4, seed=args.seed)
else:
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr,
                        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                        num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
    trainer = Trainer(model, train_dataset, None, tconf)

trainer.train()
