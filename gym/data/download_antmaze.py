import gym
import numpy as np

import collections
import pickle

import d4rl
import os

datasets = []

names = [
    'antmaze-umaze-v2',
    'antmaze-umaze-diverse-v2',
    'antmaze-medium-play-v2',
    'antmaze-medium-diverse-v2',
    'antmaze-large-play-v2',
    'antmaze-large-diverse-v2',
]

for name in names:
    env = gym.make(name)
    dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    with open(os.path.join(os.path.dirname(__file__), f'{name}.pkl'), 'wb') as f:
        pickle.dump(paths, f)
