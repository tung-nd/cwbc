import numpy as np
import torch
from scipy.stats import truncnorm

from cwbc.training.trainer import Trainer


class RVSTrainer(Trainer):

    def train_step_all(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, traj_returns = self.get_batch(self.batch_size, return_scale=False, avg_reward=self.avg_reward)
        
        state_dim = states.shape[2]
        act_dim = actions.shape[2]

        states, actions, rtg = states.reshape(-1, state_dim), actions.reshape(-1, act_dim), rtg[:,:-1].reshape(-1, 1)
        action_target = actions

        _, action_preds, _ = self.model.forward(states, rtg)

        pred_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss = pred_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/pred_loss'] = pred_loss.detach().cpu().item()

        return loss.detach().cpu().item(), traj_returns

    def get_offset(self, traj_returns, traj_len, timesteps, scale, batch_size, device):
        # augment with large offset for returns that are higher than conservative returns
        max_ids = torch.arange(batch_size)
        max_ids = max_ids[traj_returns >= self.conservative_return]
        num_augment = len(max_ids)
        if self.conservative_min is None:
            high_returns = traj_returns[max_ids] if len(max_ids) > 1 else np.array([traj_returns[max_ids]])
            min_noise = self.max_return - high_returns
            min_noise = torch.from_numpy(min_noise).reshape(num_augment, 1, 1).to(device)
        else:
            min_noise = self.conservative_min
        
        # generate offset from Uniform or Truncated Normal given min_noise and max_noise
        if self.conservative_type == 'uniform':
            max_noise = self.conservative_level
            # offset = (max_noise-min_noise) * torch.rand((num_augment, 1, 1), device=device) + min_noise
            offset = torch.zeros((num_augment, 1, 1), device=device).uniform_(0, max_noise) + min_noise
        else:
            offset = truncnorm.rvs(0.0, np.inf, loc=0.0, scale=self.conservative_std, size=num_augment)
            offset = torch.from_numpy(offset).to(device=device, dtype=next(self.model.parameters()).dtype).reshape(num_augment, 1, 1)
            offset = offset + min_noise

        if self.conservative_scale:
            offset = offset * (traj_len[max_ids] - timesteps[max_ids].unsqueeze(-1)) / traj_len[max_ids]
        if self.avg_reward:
            H = 1000.0
            offset = offset / (H - timesteps[max_ids].unsqueeze(-1))
        else:
            offset = offset / scale

        return max_ids, offset

    def train_step_percentile(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, traj_returns, scale, traj_len = self.get_batch(self.batch_size, return_scale=True, avg_reward=self.avg_reward, return_len=True)
        batch_size = actions.shape[0]
        state_dim = states.shape[2]
        act_dim = actions.shape[2]

        rtg = rtg[:,:-1]

        max_ids, offset = self.get_offset(traj_returns, traj_len, timesteps, scale, batch_size, device=rtg.device)
        aug_states, aug_actions, aug_rtg = states[max_ids], actions[max_ids], rtg[max_ids] + offset

        # normal prediction loss
        states, actions, rtg = states.reshape(-1, state_dim), actions.reshape(-1, act_dim), rtg.reshape(-1, 1)
        action_target = actions
        _, action_preds, _ = self.model.forward(states, rtg)
        pred_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # conservative loss
        if len(max_ids) > 0:
            aug_states, aug_actions, aug_rtg = aug_states.reshape(-1, state_dim), aug_actions.reshape(-1, act_dim), aug_rtg.reshape(-1, 1)
            _, aug_preds, _ = self.model.forward(aug_states, aug_rtg)
            conservative_loss = self.loss_fn(
                None, aug_preds, None,
                None, aug_actions, None,
            )
        else:
            conservative_loss = torch.Tensor([0.0]).to(pred_loss.device)

        loss = pred_loss + self.conservative_w*conservative_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/pred_loss'] = pred_loss.detach().cpu().item()
            self.diagnostics['training/conservative_loss'] = conservative_loss.detach().cpu().item()

        return loss.detach().cpu().item(), traj_returns

    def train_step(self):
        if not self.conservative_return:
            return self.train_step_all()
        else:
            return self.train_step_percentile()
