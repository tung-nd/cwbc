import numpy as np
import torch
from tqdm import tqdm
import time


class Trainer:

    def __init__(self, model, optimizer, batch_size,
        get_batch, loss_fn,
        scheduler=None, eval_fns=None,
        conservative_return=None,
        conservative_type=None,
        conservative_std=None,
        conservative_level=None,
        conservative_min=None,
        conservative_scale=None,
        conservative_w=1.0,
        max_return=None,
        avg_reward=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.conservative_return = conservative_return
        self.conservative_type = conservative_type
        self.conservative_std = conservative_std
        self.conservative_level = conservative_level
        self.conservative_min = conservative_min
        self.conservative_scale = conservative_scale
        self.conservative_w = conservative_w

        self.max_return = max_return
        self.avg_reward = avg_reward

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, logger=None, use_tqdm=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        iter_traj_returns = []
        range_steps = tqdm(range(num_steps), ascii=True) if use_tqdm else range(num_steps)
        for step in range_steps:
            train_loss, traj_returns = self.train_step()
            current_step = (iter_num-1)*num_steps + step + 1
            iter_traj_returns.append(traj_returns)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            logger.info('=' * 80)
            logger.info(f'Iteration {iter_num}')
            for k, v in logs.items():
                logger.info(f'{k}: {v}')

        return logs, np.array(iter_traj_returns)

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
