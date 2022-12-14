import torch
import torch.nn as nn
from torch.nn import functional as F

class RVS(nn.Module):

    """
    Simple MLP that predicts next action a from past states s and reward-to-go r.
    """

    def __init__(self, n_embed, act_dim, hidden_size, n_layer, dropout=0.1, max_length=None):
        super().__init__()

        self.n_embed = n_embed
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, self.n_embed)
        )

        self.ret_emb = nn.Sequential(nn.Linear(1, self.n_embed))

        layers = [nn.LayerNorm(self.n_embed*2), nn.Linear(self.n_embed*2, hidden_size)] # input is concat of state and rtg
        for _ in range(n_layer-1):
            layers.extend([
                # nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
        ])

        self.model = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions, rtg, reduction='mean'):
        state_emb = self.state_encoder(states)
        rtg_emb = self.ret_emb(rtg)
        inp = torch.cat((state_emb, rtg_emb), dim=-1)
        # inp = torch.cat((state_emb, rtg), dim=-1)
        logits = self.model(inp)
        if actions is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, actions.reshape(-1), reduction=reduction)
        return logits, loss

    def get_action(self, states, rtg):
        logits, _ = self.forward(states, None, rtg)
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        return ix
