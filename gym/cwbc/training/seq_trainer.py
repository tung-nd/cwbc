import numpy as np
import torch
from scipy.stats import truncnorm

from cwbc.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step_all(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, traj_returns, scale = self.get_batch(self.batch_size, return_scale=True, avg_reward=self.avg_reward)
        batch_size = actions.shape[0]

        rtg_ = rtg[:,:-1]

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states, actions, rewards, rtg_, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        pred_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss = pred_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/pred_loss'] = pred_loss.detach().cpu().item()

        return loss.detach().cpu().item(), traj_returns

    def train_step_percentile(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, traj_returns, scale, traj_len = self.get_batch(self.batch_size, return_scale=True, avg_reward=self.avg_reward, return_len=True)
        batch_size = actions.shape[0]

        rtg_ = rtg[:,:-1]
        action_target = torch.clone(actions)

        # normal prediction loss
        _, action_preds, _ = self.model.forward(
            states, actions, rewards, rtg_, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        pred_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # conservative loss
        # augment with large offset for returns that are higher than conservative returns
        max_ids = torch.arange(batch_size)
        max_ids = max_ids[traj_returns >= self.conservative_return]
        num_augment = len(max_ids)
        if num_augment > 0:
            if self.conservative_min is None:
                high_returns = traj_returns[max_ids] if len(max_ids) > 1 else np.array([traj_returns[max_ids]])
                min_noise = self.max_return - high_returns
                min_noise = torch.from_numpy(min_noise).reshape(num_augment, 1, 1).to(rtg_.device)
            else:
                min_noise = self.conservative_min

            # generate offset from Uniform or Truncated Normal given min_noise and max_noise
            if self.conservative_type == 'uniform':
                max_noise = self.conservative_level
                offset = (max_noise-min_noise) * torch.rand((num_augment, 1, 1), device=rtg_.device) + min_noise
            else:
                offset = truncnorm.rvs(0.0, np.inf, loc=0.0, scale=self.conservative_std, size=num_augment)
                offset = torch.from_numpy(offset).to(device=rtg_.device, dtype=rtg_.dtype).reshape(num_augment, 1, 1)
                offset = offset + min_noise
            
            if self.conservative_scale:
                offset = offset * (traj_len[max_ids] - timesteps[max_ids].unsqueeze(-1)) / traj_len[max_ids]
            if self.avg_reward:
                H = 1000.0
                offset = offset / (H - timesteps[max_ids].unsqueeze(-1))
            else:
                offset = offset / scale

            _, aug_action_preds, _ = self.model.forward(
                states[max_ids], actions[max_ids], rewards[max_ids], rtg_[max_ids] + offset, timesteps[max_ids], attention_mask=attention_mask[max_ids],
            )

            att_mask_chosen = attention_mask[max_ids]
            aug_a_target = actions[max_ids].reshape(-1, act_dim)[att_mask_chosen.reshape(-1) > 0]
            aug_action_preds = aug_action_preds.reshape(-1, act_dim)[att_mask_chosen.reshape(-1) > 0]
            conservative_loss = self.loss_fn(
                None, aug_action_preds, None,
                None, aug_a_target, None
            )
        else:
            conservative_loss = torch.Tensor([0.0]).to(action_preds.device)

        loss = pred_loss + self.conservative_w*conservative_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
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
