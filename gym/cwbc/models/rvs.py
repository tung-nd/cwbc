import numpy as np
import torch
import torch.nn as nn

from cwbc.models.model import TrajectoryModel


class RVS(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s and reward-to-go r.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=None):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        layers = [nn.Linear(self.state_dim+1, hidden_size)] # input is concat of state and rtg
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, rtg):
        inp = torch.cat((states, rtg), dim=-1)
        model_dtype = next(self.parameters()).dtype
        actions = self.model(inp.to(dtype=model_dtype))
        return None, actions, None

    def get_action(self, states, rtg):
        model_dtype = next(self.parameters()).dtype
        _, actions, _ = self.forward(states.to(dtype=model_dtype), rtg.to(dtype=model_dtype))
        return actions[0]
