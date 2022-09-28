import os.path as osp
import os
import time
import gym
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from attrdict import AttrDict
from pathlib import Path
import matplotlib.pyplot as plt
import random
import json
from functools import partial
import math

from utils import load_dataset, create_model, LOSS_DICT, TRAINER_DICT, EXPERT_RETURN, ANTMAZE_MAP

from cwbc.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_rvs
from cwbc.utils.logs import get_logger, draw_hist
from cwbc.utils.data import reweight_bin


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def avg_cumsum(x, gamma, start_timestep, max_ep_len):
    cumsum = discount_cumsum(x, gamma)
    t = np.arange(len(x)) + start_timestep
    return cumsum / (1 + max_ep_len - t) # RvS paper

def get_env(env_name, dataset, seed):
    import d4rl
    if f'{env_name}-{dataset}-v2' in ANTMAZE_MAP:
        name = ANTMAZE_MAP[f'{env_name}-{dataset}-v2']
    else:
        name = f'{env_name}-{dataset}-v2'
    env = gym.make(name)
    if env_name == 'hopper':
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        max_ep_len = 1000
        env_targets = [5000, 2500, 1000]
        scale = 1000.
    elif env_name in ['umaze', 'medium', 'large']:
        max_ep_len = 1000
        env_targets = [1.0, 0.0]
        scale = 1.0
    else:
        raise NotImplementedError

    env.seed(seed)
    if hasattr(env.env, "wrapped_env"):
        env.env.wrapped_env.seed(seed)
    elif hasattr(env.env, "seed"):
        env.env.seed(seed)
    else:
        pass
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, max_ep_len, env_targets, scale, state_dim, act_dim


def get_batch(
    max_len, num_trajectories,
    p_sample, returns, percentile, sorted_inds,
    trajectories, max_ep_len, state_dim,
    state_mean, state_std, act_dim,
    scale, device, batch_size,
    batch_inds=None, sis=None,
    return_scale=False, avg_reward=False,
    return_len=False
):
    if batch_inds is None:
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample
        )

    traj_returns = returns[sorted_inds[batch_inds]]

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    traj_len = []
    if avg_reward:
        scale = 1.
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        si = random.randint(0, traj['rewards'].shape[0] - 1) if sis is None else sis[i]

        # get sequences from dataset
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        traj_len.append(len(traj['rewards']))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
        if not avg_reward:
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        else:
            rtg.append(avg_cumsum(traj['rewards'][si:], gamma=1., start_timestep=si, max_ep_len=max_ep_len)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))


    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
    traj_len = torch.from_numpy(np.array(traj_len)).to(device=device).reshape(-1, 1, 1)

    to_return = (s, a, r, d, rtg, timesteps, mask, traj_returns)
    if return_scale:
        to_return = to_return + (scale,)
    if return_len:
        to_return = to_return + (traj_len,)
    return to_return

def train(
        exp_prefix,
        variant,
):
    if osp.exists(variant['root'] + '/ckpt.tar'):
        variant['resume'] = True
    else:
        os.makedirs(variant['root'], exist_ok=True)
    with open(osp.join(variant['root'], 'args.yaml'), 'wb') as f:
        OmegaConf.save(config=variant, f=f.name)

    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    expid = variant['expid']
    exp_prefix = f'{exp_prefix}-{env_name}-{dataset}-{model_type}-{expid}'

    # get environment
    env, max_ep_len, env_targets, scale, state_dim, act_dim = get_env(env_name, dataset, seed=variant['seed'])
    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    # load dataset
    mode = variant.get('mode', 'normal')
    trajectories, traj_lens, returns, state_mean, state_std = load_dataset(variant['dataset_path'], mode)

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    sorted_returns = returns[sorted_inds]

    # save histogram of dataset rtg
    nbins = variant['bins']
    draw_hist(sorted_returns, env_name, dataset, nbins, osp.join(variant['root'], 'hist_dataset'))

    if variant['reweight_rtg']:
        print ('Reweighting dataset')
        # MINs-like reweighting
        freq, edges = np.histogram(sorted_returns, bins=nbins)
        orig_hist = freq / np.sum(freq)
        if variant['reweight_uniform']:
            reweighted_hist = np.ones(len(orig_hist)) * 1 / len(orig_hist)
        else:
            reweighted_hist = reweight_bin(orig_hist, edges, sorted_returns, variant['percentile'], variant['lamb'])
        ids = np.digitize(sorted_returns, edges, right=True) - 1
        ids[0] = ids[0] + 1
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
        plt.savefig(osp.join(variant['root'], 'reweighted_hist'))
        plt.clf()
        plt.close()
    else:
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    model = create_model(model_type, state_dim, act_dim, K, max_ep_len, variant)
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if variant['resume']:
        ckpt = torch.load(osp.join(variant['root'], 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_iter = ckpt.iter
    else:
        logfilename = osp.join(variant['root'], 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_iter = 1

    logger = get_logger(logfilename)

    if not variant['resume']:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    partial_get_batch = partial(get_batch,
        K, num_trajectories, p_sample,
        returns, variant['percentile'], sorted_inds, trajectories, max_ep_len,
        state_dim, state_mean, state_std, act_dim,
        scale, device
    )

    loss_fn = LOSS_DICT[model_type]
    trainer_class = TRAINER_DICT[model_type]
    conservative_return = None
    percentile = variant['conservative_percentile']
    if percentile is not None:
        conservative_return = sorted_returns[int(percentile * len(sorted_returns))]
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=partial_get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=None, # do not evaluate during training
        conservative_return=conservative_return,
        conservative_type=variant['conservative_type'],
        conservative_std=variant['conservative_std'],
        conservative_level=variant['conservative_level'],
        conservative_min=variant['conservative_min'],
        conservative_scale=variant['conservative_scale'],
        conservative_w=variant['conservative_w'],
        max_return=sorted_returns[-1],
        avg_reward=variant['avg_reward']
    )

    all_traj_returns = []
    for iter in range(start_iter, variant['max_iters']+1):
        outputs, iter_traj_returns = trainer.train_iteration(
            num_steps=variant['num_steps_per_iter'], iter_num=iter,
            print_logs=True, logger=logger, use_tqdm=variant['tqdm'])
        all_traj_returns.append(iter_traj_returns)
        if iter % variant['save_freq'] == 0 or iter == variant['max_iters']:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.iter = iter + 1
            ckpt.all_traj_returns = all_traj_returns
            torch.save(ckpt, osp.join(variant['root'], 'ckpt.tar'))
    all_traj_returns = np.array(all_traj_returns).flatten()
    draw_hist(all_traj_returns, env_name, dataset, nbins, osp.join(variant['root'], 'hist_training'))

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

def evaluate(variant, env_targets):
    # import d4rl
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    K = variant['K']
    num_eval_episodes = variant['num_eval_episodes']

    # get environment
    env, max_ep_len, _, scale, state_dim, act_dim = get_env(env_name, dataset, seed=variant['seed'])
    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    # load dataset
    mode = variant.get('mode', 'normal')
    _, _, _, state_mean, state_std = load_dataset(variant['dataset_path'], mode)

    # load model
    model = create_model(model_type, state_dim, act_dim, K, max_ep_len, variant)
    model = model.to(device=device)
    ckpt_path = variant['root']
    ckpt = torch.load(osp.join(ckpt_path, 'ckpt.tar'))
    model.load_state_dict(ckpt.model)

    # create logger
    logfilename = osp.join(variant['root'], 'eval.log')
    logger = get_logger(logfilename, mode='w')

    eval_results = {}

    for target in env_targets:
        target = float(target)
        eval_results[target] = {}
        returns, unom_returns, lengths = [], [], []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                if model_type in ['dt', 'dtmi', 'dtstoc', 'dtmia', 'dtreturns', 'dtmiseq']:
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                elif model_type == 'rvs':
                    ret, length = evaluate_episode_rvs(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        avg_reward=variant['avg_reward']
                    )
                else:
                    ret, length = evaluate_episode(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
            returns.append(env.get_normalized_score(ret)*100)
            unom_returns.append(ret)
            lengths.append(length)

        logger.info(f'target_{target}_return_mean: : {np.mean(returns)}')
        eval_results[target]['return_mean'] = np.mean(returns)
        logger.info(f'target_{target}_return_std: {np.std(returns)}')
        eval_results[target]['return_std'] = np.std(returns)

        logger.info(f'target_{target}_unormalized_return_mean: : {np.mean(unom_returns)}')
        eval_results[target]['unormalized_return_mean'] = np.mean(unom_returns)
        logger.info(f'target_{target}_unormalized_return_std: {np.std(unom_returns)}')
        eval_results[target]['unormalized_return_std'] = np.std(unom_returns)

        logger.info(f'target_{target}_length_mean: {np.mean(lengths)}')
        eval_results[target]['len_mean'] = np.mean(lengths)
        logger.info(f'target_{target}_length_std: {np.std(lengths)}')
        eval_results[target]['len_std'] = np.std(lengths)
        logger.info("=" * 80)

    with open(osp.join(variant['root'], 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f)

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    seed = cfg['seed']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cfg['root'] = str(Path.cwd())

    # for antmaze exps
    if 'play' == cfg['dataset'] or 'diverse' == cfg['dataset']:
        env_name = cfg['env']
        dataset_type = cfg['dataset']
        name = ANTMAZE_MAP[f'{env_name}-{dataset_type}-v2']
        dataset_dir = cfg['dataset_dir']
        cfg['dataset_path'] = f'{dataset_dir}/{name}.pkl'


    if cfg['run_mode'] == 'train':
        cfg['conservative_level'] = math.sqrt(12*(cfg['conservative_std']**2))
        if cfg['resume_dir']:
            cfg['root'] = cfg['resume_dir']
            cfg['resume'] = True
        train('gym', cfg)
        # evaluate after training
        if 'play' == cfg['dataset'] or 'diverse' == cfg['dataset']:
            targets = np.array([1.0])
        else:
            cfg['eval_max_target'] = EXPERT_RETURN[cfg['env']]
            targets = np.linspace(1000, cfg['eval_expert_factor']*EXPERT_RETURN[cfg['env']], num=cfg['eval_intervals'])
            targets = np.array(list(targets) + [EXPERT_RETURN[cfg['env']]])
        
        evaluate(cfg, targets)

if __name__ == '__main__':
    main()
