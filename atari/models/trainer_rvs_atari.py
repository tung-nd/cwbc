import os
import numpy as np
import torch
import math

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

import atari_py
from collections import deque
import random
import cv2
import torch
import json

from utils import get_logger, EXPERT_RETURN, RANDOM_RETURN, MAX_CONDITION_RETURN

class RVSTrainer:

    def __init__(
        self, root, model, avg_reward, game, train_dataset, max_timesteps, min_rtg,
        max_epochs, batch_size, lr,
        conservative_level, conservative_return, conservative_w, max_return,
        reweight_rtg,
        betas = (0.9, 0.95), decay=0.1,
        lr_decay=False, warmup_tokens=375e6,
        final_tokens=260e9, num_workers=0, seed=0
    ):
        # TODO: lr scheduler
        self.root = root
        self.model = model
        self.avg_reward = avg_reward
        self.game = game
        self.train_dataset = train_dataset
        self.max_timesteps = max_timesteps
        self.min_rtg = min_rtg
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.conservative_return = conservative_return
        self.conservative_level = conservative_level
        self.conservative_w = conservative_w
        self.reweight_rtg = reweight_rtg
        self.max_return = max_return
        self.lr = lr
        self.betas = betas
        self.decay = decay
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.num_workers = num_workers
        self.seed = seed

        self.random_return = RANDOM_RETURN[game]
        self.expert_return = EXPERT_RETURN[game]

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.to(self.device)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, betas=self.betas)
        return optimizer

    def get_offset(self, traj_returns, timesteps, device):
        # augment with large offset for returns that are higher than conservative returns
        max_ids = torch.arange(traj_returns.shape[0])
        max_ids = max_ids[traj_returns >= self.conservative_return]
        num_augment = len(max_ids)

        if num_augment == 0:
            return max_ids, None

        high_returns = traj_returns[max_ids] if len(max_ids) > 1 else torch.from_numpy(np.array([traj_returns[max_ids]]))
        min_noise = self.max_return - high_returns
        min_noise = min_noise.reshape(num_augment, 1, 1).to(device)
        
        max_noise = self.conservative_level
        offset = torch.zeros((num_augment, 1, 1), device=device).uniform_(0, max_noise) + min_noise
        offset = offset.float()

        if self.avg_reward:
            offset = offset / (self.max_timesteps - timesteps[max_ids] + 1)

        return max_ids, offset

    def train(self):
        model = self.model
        optimizer = self.configure_optimizers()

        all_traj_returns = []

        def run_epoch(epoch_num=0):
            model.train(True)
            data = self.train_dataset
            loader = DataLoader(
                data, shuffle=True, pin_memory=True,
                batch_size=self.batch_size, num_workers=self.num_workers
            )

            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (states, actions, rtgs, timesteps, traj_returns) in pbar:
                all_traj_returns.extend(list(traj_returns.numpy().flatten()))
                # place data on the correct device
                states = states.to(self.device)
                actions = actions.to(self.device)
                rtgs = rtgs.to(self.device)
                timesteps = timesteps.to(self.device)

                if self.conservative_return is not None:
                    max_ids, offset = self.get_offset(traj_returns, timesteps, self.device)
                    if len(max_ids) > 0:
                        aug_states, aug_actions, aug_rtgs = states[max_ids], actions[max_ids], rtgs[max_ids] + offset

                # place data on the correct device
                states = states.flatten(0, 1)
                actions = actions.flatten(0, 1)
                rtgs = rtgs.flatten(0, 1)

                # forward the model
                _, pred_loss = model(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(), actions, rtgs, reduction='none')
                pred_loss = torch.mean(pred_loss)

                # conservative loss
                if self.conservative_return is not None and len(max_ids) > 0:
                    aug_states = aug_states.flatten(0, 1)
                    aug_actions = aug_actions.flatten(0, 1)
                    aug_rtgs = aug_rtgs.flatten(0, 1)
                    _, conservative_loss = model(
                        aug_states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(),
                        aug_actions, aug_rtgs, reduction='none'
                    )
                    conservative_loss = torch.mean(conservative_loss)
                    loss = pred_loss + self.conservative_w * conservative_loss

                else:
                    conservative_loss = torch.Tensor([0.0]).to(self.device)
                    loss = pred_loss
                
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                optimizer.step()

                if self.lr_decay:
                    self.tokens += (actions >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < self.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - self.warmup_tokens) / float(max(1, self.final_tokens - self.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.lr * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.lr

                pbar.set_description(
                    f"epoch {epoch_num} iter {it}: pred loss {pred_loss.item():.3f}, conservative loss {conservative_loss.item():.3f}. lr {lr:e}"
                )

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(self.max_epochs):
            run_epoch(epoch)
            
        rtgs = np.linspace(start=RANDOM_RETURN[self.game], stop=MAX_CONDITION_RETURN[self.game], num=20)
        eval_returns = self.get_multiple_returns(rtgs)

        return all_traj_returns

    def normalize_return(self, ret):
        return 100. * (ret - self.random_return) / (self.expert_return - self.random_return)

    def get_multiple_returns(self, list_ret):
        self.model.train(False)

        returns = []
        eval_results = {}
        for ret in list_ret:
            r, eval_results = self.get_returns(ret, eval_results)
            returns.append(r)

        with open(os.path.join(self.root, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f)

        self.model.train(True)

        return returns

    def get_returns(self, ret, eval_results):
        args=Args(self.game.lower(), self.seed)
        env = Env(args)
        env.eval()

        logfilename = os.path.join(self.root, 'eval.log')
        logger = get_logger(logfilename, mode='a')

        T_rewards = []
        normalized_rewards = []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0)
            rtgs = torch.tensor([ret], dtype=torch.long).to(self.device).unsqueeze(0)
            if self.avg_reward:
                avg_rtgs = rtgs / (1 + self.max_timesteps)
            else:
                avg_rtgs = rtgs
            sampled_action = self.model.get_action(state.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(), avg_rtgs.float())

            j = 0
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0, -1]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    normalized_rewards.append(self.normalize_return(reward_sum))
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                rtgs = rtgs - reward # TODO: rvs update

                timesteps = min(j, self.max_timesteps)
                if self.avg_reward:
                    avg_rtgs = rtgs / (1 + self.max_timesteps - timesteps)
                else:
                    avg_rtgs = rtgs

                if self.game != 'Pong': # reset rtg to min_rtg if negative, except for Pong
                    avg_rtgs = torch.maximum(avg_rtgs, torch.ones_like(avg_rtgs) * self.min_rtg)

                sampled_action = self.model.get_action(state.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(), avg_rtgs.float())

        env.close()

        T_rewards = np.array(T_rewards)
        mean_return = np.mean(T_rewards)
        std_return = np.std(T_rewards)

        normalized_rewards = np.array(normalized_rewards)
        norm_mean_return = np.mean(normalized_rewards)
        norm_std_return = np.std(normalized_rewards)

        eval_results[ret] = {}

        logger.info(f'target_{ret}_unormalized_return_mean: : {mean_return}')
        eval_results[ret]['unormalized_return_mean'] = mean_return
        logger.info(f'target_{ret}_unormalized_return_std: {std_return}')
        eval_results[ret]['unormalized_return_std'] = std_return

        logger.info(f'target_{ret}_return_mean: : {norm_mean_return}')
        eval_results[ret]['return_mean'] = norm_mean_return
        logger.info(f'target_{ret}_return_std: {norm_std_return}')
        eval_results[ret]['return_std'] = norm_std_return

        logger.info("=" * 70)

        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()

        return mean_return, eval_results


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
