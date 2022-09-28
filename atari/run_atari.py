import os
from models.utils import set_seed
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from models.model_atari import GPT, GPTConfig
from models.trainer_atari import Trainer, TrainerConfig
from models.rvs_atari import RVS
from models.trainer_rvs_atari import RVSTrainer
import random
import torch
import pickle
import argparse
from utils import reweight_bin, draw_hist
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()

# general
parser.add_argument('--root', type=str, default='exp')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--data_dir', type=str, default='./data_5e5/')

# model
parser.add_argument('--model_type', type=str, default='rvs')
parser.add_argument('--avg_reward', action='store_true', default=False)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--n_embed', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)

# training
parser.add_argument('--lr', type=float, default=6e-4)
parser.add_argument('--decay', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)

# conservative 
parser.add_argument('--conservative_percentile', type=float, default=None)
parser.add_argument('--conservative_std', type=float, default=50.0)
parser.add_argument('--conservative_w', type=float, default=0.1)

# reweight
parser.add_argument('--reweight_rtg', action='store_true', default=False)
parser.add_argument('--bins', type=int, default=20)
parser.add_argument('--lamb', type=float, default=0.1)
parser.add_argument('--percentile', type=float, default=0.5)

args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, rtgs, timesteps, returns, p_sample, sorted_inds):        
        self.block_size = block_size
        self.vocab_size = max(np.concatenate(actions, axis=0)) + 1
        self.data = data
        self.actions = actions
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.returns = returns
        self.p_sample = p_sample
        self.sorted_inds = sorted_inds
    
    def __len__(self):
        return len(np.concatenate(self.actions, axis=0)) - self.block_size

    def __getitem__(self, _):
        block_size = self.block_size // 3
        while True:
            idx = np.random.choice(np.arange(len(self.returns)), p=p_sample)
            idx = self.sorted_inds[idx]
            if self.data[idx].shape[0] > block_size:
                break
        si = random.randint(0, self.data[idx].shape[0] - block_size)
        states = torch.tensor(self.data[idx][si:si+block_size], dtype=torch.float32).reshape(block_size, -1)
        states = states / 255.
        actions = torch.tensor(self.actions[idx][si:si+block_size], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx][si:si+block_size], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx][si:si+block_size], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps, self.returns[idx]

root = os.path.join(args.root, f'{args.game.lower()}_{str(args.seed)}')
os.makedirs(root, exist_ok=True)

with open(os.path.join(args.data_dir, f'{args.game.lower()}.pkl'), "rb") as f:
    data = pickle.load(f)
traj_obs, traj_actions, traj_returns, traj_rtgs, traj_timesteps, max_timesteps = data['obs'], data['actions'], data['returns'], data['rtgs'], data['timesteps'], data['max_timesteps']
min_rtg = min([rtg.min() for rtg in traj_rtgs])
draw_hist(traj_returns, args.game, args.bins, os.path.join(root, 'hist_dataset'))

sorted_inds = np.argsort(traj_returns)
sorted_returns = traj_returns[sorted_inds]

if args.reweight_rtg:
    freq, edges = np.histogram(sorted_returns, bins=args.bins)
    orig_hist = freq / np.sum(freq)
    reweighted_hist = reweight_bin(orig_hist, edges, sorted_returns, args.percentile, args.lamb)
    ids = np.digitize(sorted_returns, edges, right=False) - 1
    ids[-1] = ids[-1] - 1

    border_ids = (ids == args.bins)
    ids[border_ids] = ids[border_ids] - 1

    p_sample = np.zeros(len(ids))
    for i in range(len(p_sample)):
        p_sample[i] = reweighted_hist[ids[i]] / freq[ids[i]]
    p_sample = p_sample / np.sum(p_sample)

    # plot reweighted histogram
    _, ax = plt.subplots()
    ax.bar(edges[:-1], reweighted_hist, width=np.diff(edges), edgecolor="black", align="edge")
    ax.set(xlabel='Returns')
    ax.set(ylabel='Density')
    plt.tight_layout()
    plt.savefig(os.path.join(root, 'reweighted_hist'))
    plt.clf()
    plt.close()
else:
    p_sample = np.ones(len(traj_returns))
    p_sample = p_sample / np.sum(p_sample)

train_dataset = StateActionReturnDataset(traj_obs, args.context_length*3, traj_actions, traj_rtgs, traj_timesteps, traj_returns, p_sample, sorted_inds)

epochs = args.epochs

if args.model_type == 'rvs':
    model = RVS(n_embed=args.n_embed, act_dim=train_dataset.vocab_size, hidden_size=args.hidden_size, n_layer=args.n_layer, dropout=args.dropout)
    sorted_returns = np.sort(traj_returns)
    
    if args.conservative_percentile is not None:
        conservative_level = math.sqrt(12*(args.conservative_std**2))
        conservative_return = sorted_returns[int(args.conservative_percentile * len(sorted_returns))]
        conservative_w = args.conservative_w
        print ('conservative return', conservative_return)
        print ('conservative level', conservative_level)
    else:
        conservative_level = None
        conservative_return = None
        conservative_w = None

    with open(os.path.join(root, 'args.yaml'), 'wb') as f:
        OmegaConf.save(config=vars(args), f=f.name)

    trainer = RVSTrainer(
        root, model, args.avg_reward, args.game, train_dataset, max_timesteps, min_rtg, epochs, args.batch_size, lr=args.lr, decay=args.decay,
        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
        conservative_return=conservative_return, conservative_level=conservative_level, conservative_w=conservative_w,
        reweight_rtg=args.reweight_rtg,
        max_return=sorted_returns[-1], num_workers=4, seed=args.seed)

    all_traj_returns = trainer.train()
    all_traj_returns = np.array(all_traj_returns).flatten()
    draw_hist(all_traj_returns, args.game, args.bins, os.path.join(root, 'hist_training'))
else:
    if args.model_type == 'dt':
        model_type = 'reward_conditioned'
    elif args.model_type == 'bc':
        model_type = 'naive'
    else:
        raise NotImplementedError
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max_timesteps)
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr,
                        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                        num_workers=4, seed=args.seed, model_type=model_type, game=args.game, max_timestep=max_timesteps)
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()
