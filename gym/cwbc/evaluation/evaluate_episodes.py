import numpy as np
import torch

numpy_to_torch_dtype_dict = {
    np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rvs(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        avg_reward=True
):

    model.eval()
    model.to(device=device)
    model_dtype = next(model.parameters()).dtype

    state_mean = torch.from_numpy(state_mean).to(device=device, dtype=model_dtype)
    state_std = torch.from_numpy(state_std).to(device=device, dtype=model_dtype)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    state = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=model_dtype)

    target_return = torch.from_numpy(np.array(target_return)).reshape(1, 1).to(device=device, dtype=model_dtype)
    if avg_reward:
        target_return = target_return / max_ep_len
    else:
        target_return = target_return / scale

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        
        # predict action given current state and rtg
        action = model.get_action(
            (state.to(dtype=model_dtype) - state_mean) / state_std,
            target_return.to(dtype=model_dtype)
        )
        action = action.detach().cpu().numpy()

        # apply action to the environment
        state, reward, done, _ = env.step(action)
        # reward = reward.astype(torch_to_numpy_dtype_dict[model_dtype])
        # reward = np.float32(reward)

        # update state and rtg
        state = torch.from_numpy(state).to(device=device, dtype=model_dtype).reshape(1, state_dim)
        if avg_reward:
            target_return = ((max_ep_len-t)*target_return - reward) / (max_ep_len - t - 1)
        else:
            target_return = target_return - reward/scale
        target_return = target_return.to(dtype=model_dtype)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        init_state=None,
        return_actions=False,
        avg_reward=False
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset() if init_state is None else np.copy(init_state)
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    if avg_reward:
        target_return = target_return / max_ep_len
    else:
        target_return = target_return / scale
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            if avg_reward:
                pred_return = ((max_ep_len-t)*target_return[0,-1] - reward) / (max_ep_len - t - 1)
            else:
                pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    if not return_actions:
        return episode_return, episode_length
    else:
        return episode_return, episode_length, actions, target_return, rewards
