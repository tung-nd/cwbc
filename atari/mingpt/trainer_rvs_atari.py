from turtle import back
import numpy as np
import torch
from scipy.stats import truncnorm
import torch.nn as nn
import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image

class RVSTrainer:

    def __init__(
        self, model, game, train_dataset,
        max_epochs, batch_size, lr,
        conservative_level, conservative_return, conservative_w, max_return,
        betas = (0.9, 0.95), decay=0.1,
        lr_decay=False, warmup_tokens=375e6,
        final_tokens=260e9, num_workers=0, seed=0
    ):
        # TODO: lr scheduler
        self.model = model
        self.game = game
        self.train_dataset = train_dataset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.conservative_return = conservative_return
        self.conservative_level = conservative_level
        self.conservative_w = conservative_w
        self.max_return = max_return
        self.lr = lr
        self.betas = betas
        self.decay = decay
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.num_workers = num_workers
        self.seed = seed

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.to(self.device)

    def get_offset(self, traj_returns, device):
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

        return max_ids, offset

    def train(self):
        model = self.model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.decay,
        )

        def run_epoch(epoch_num=0):
            model.train(True)
            data = self.train_dataset
            loader = DataLoader(
                data, shuffle=True, pin_memory=True,
                batch_size=self.batch_size, num_workers=self.num_workers
            )

            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (states, actions, rtgs, _, traj_returns) in pbar:
                # place data on the correct device
                states = states.to(self.device)
                actions = actions.to(self.device)
                rtgs = rtgs.to(self.device)

                if self.conservative_return is not None:
                    max_ids, offset = self.get_offset(traj_returns, self.device)
                    if len(max_ids) > 0:
                        aug_states, aug_actions, aug_rtgs = states[max_ids], actions[max_ids], rtgs[max_ids] + offset

                # place data on the correct device
                states = states.flatten(0, 1)
                actions = actions.flatten(0, 1)
                rtgs = rtgs.flatten(0, 1)

                # forward the model
                _, pred_loss = model(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(), actions, rtgs, reduction='mean')

                # conservative loss
                if self.conservative_return is not None and len(max_ids) > 0:
                    aug_states = aug_states.flatten(0, 1)
                    aug_actions = aug_actions.flatten(0, 1)
                    aug_rtgs = aug_rtgs.flatten(0, 1)
                    _, conservative_loss = model(
                        aug_states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(),
                        aug_actions, aug_rtgs, reduction='mean'
                    )
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
            if self.game == 'Breakout':
                eval_return = self.get_returns(90)
            elif self.game == 'Seaquest':
                rtgs = np.linspace(start=150, stop=1150, num=11)
                eval_return = self.get_multiple_returns(rtgs)
                # eval_return = self.get_returns(1150)
            elif self.game == 'Qbert':
                eval_return = self.get_returns(14000)
            elif self.game == 'Pong':
                eval_return = self.get_returns(20)
            else:
                raise NotImplementedError()

    def get_multiple_returns(self, list_ret):
        returns = []
        for ret in list_ret:
            returns.append(self.get_returns(ret))
        return returns

    def get_returns(self, ret):
        self.model.train(False)
        args=Args(self.game.lower(), self.seed)
        env = Env(args)
        env.eval()

        T_rewards = []
        done = True
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0)
            rtgs = torch.tensor([ret], dtype=torch.long).to(self.device).unsqueeze(0)
            sampled_action = self.model.get_action(state.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(), rtgs.float())

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
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                rtgs = rtgs - reward # TODO: rvs update

                sampled_action = self.model.get_action(state.reshape(-1, 4, 84, 84).type(torch.float32).contiguous(), rtgs.float())

        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return eval_return


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
