import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size=50, n_layers=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, g, x, h):
        # x: (bs, seq_len, N, input_size)
        bs, seq_len, N, _ = x.size()
        x = x.permute(1, 0, 2, 3).reshape(seq_len, bs * N, -1)

        y, h = self.gru(x, h)
        y = self.output(y)

        y = y.reshape(seq_len, bs, N, -1).permute(1, 0, 2, 3)
        return y, h

    def default_h(self, bs):
        return torch.zeros(self.n_layers, bs * self.num_nodes,
                           self.hidden_size)
