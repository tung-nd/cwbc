import numpy as np
from fixed_replay_buffer import FixedReplayBuffer

from utils import EXPERT_RETURN, RANDOM_RETURN

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer, avg_reward=False, save_dir=None):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    obss = np.array(obss)
    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    max_rtg = max(rtg)
    print('max rtg is %d' % max_rtg)
    norm_max_rtg = 100. * (max_rtg - RANDOM_RETURN[game]) / (EXPERT_RETURN[game] - RANDOM_RETURN[game])
    print('max normalized rtg is %f' % norm_max_rtg)

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions), dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i] = np.arange(i - start_index)
        start_index = i
    print('max timestep is %d' % max(timesteps))

    if avg_reward:
        rtg = rtg / (1 + max(timesteps) - timesteps)

    ### create list of trajectories
    traj_obs, traj_actions, traj_rtgs, traj_timesteps = [], [], [], []
    start_index = 0
    for k, i in enumerate(done_idxs):
        i = int(i)
        traj_obs.append(obss[start_index:i])
        traj_actions.append(actions[start_index:i])
        traj_rtgs.append(rtg[start_index:i])
        traj_timesteps.append(timesteps[start_index:i])
        start_index = i

    if save_dir is not None:
        data = {
            'obs': traj_obs,
            'actions': traj_actions,
            'rtgs': traj_rtgs,
            'timesteps': traj_timesteps,
            'returns': returns[:-1],
            'max_timesteps': max(timesteps)
        }
        import os
        os.makedirs(save_dir, exist_ok=True)
        import pickle
        with open(os.path.join(save_dir, f'{game.lower()}.pkl'), "wb") as f:
            pickle.dump(data, f)

    return traj_obs, traj_actions, returns[:-1], traj_rtgs, traj_timesteps, max(timesteps)

import argparse
parser = argparse.ArgumentParser()

# general
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--avg_reward', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default='./data_5e5')

args = parser.parse_args()

traj_obs, traj_actions, traj_returns, traj_rtgs, traj_timesteps, max_timesteps = create_dataset(
    num_buffers=args.num_buffers, num_steps=args.num_steps,
    game=args.game, data_dir_prefix=args.data_dir_prefix,
    trajectories_per_buffer=args.trajectories_per_buffer,
    avg_reward=args.avg_reward, save_dir=args.save_dir
)