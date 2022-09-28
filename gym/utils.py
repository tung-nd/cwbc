import pickle
import numpy as np
import torch


from cwbc.models.decision_transformer import DecisionTransformer
from cwbc.models.mlp_bc import MLPBCModel
from cwbc.models.rvs import RVS

from cwbc.training.rvs_trainer import RVSTrainer
from cwbc.training.act_trainer import ActTrainer
from cwbc.training.seq_trainer import SequenceTrainer

MODEL_DICT = {
    "dt": DecisionTransformer,
    "bc": MLPBCModel,
    "rvs": RVS
}
LOSS_DICT = {
    "dt": lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
    "bc": lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
    "rvs": lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
}
TRAINER_DICT = {
    "dt": SequenceTrainer,
    "bc": ActTrainer,
    "rvs": RVSTrainer
}
EXPERT_RETURN = {"walker2d": 4600, "hopper": 3200, "halfcheetah": 12000}

ANTMAZE_MAP = {
    'umaze-play-v2': 'antmaze-umaze-v2',
    'umaze-diverse-v2': 'antmaze-umaze-diverse-v2',
    'medium-play-v2': 'antmaze-medium-play-v2',
    'medium-diverse-v2': 'antmaze-medium-diverse-v2',
    'large-play-v2': 'antmaze-large-play-v2',
    'large-diverse-v2': 'antmaze-large-diverse-v2'
}


def load_dataset(dataset_path, mode):
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    return trajectories, traj_lens, returns, state_mean, state_std


def create_model(model_type, state_dim, act_dim, K, max_ep_len, variant):
    class_model = MODEL_DICT[model_type]
    if model_type in ["dt"]:
        model = class_model(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            concat_state_rtg=variant["concat_state_rtg"],
            avg_reward=variant["avg_reward"]
        )
    elif model_type in ["bc", "rvs"]:
        model = class_model(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError

    return model
