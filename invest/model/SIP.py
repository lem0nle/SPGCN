import torch
import torch.nn as nn
import torch.nn.functional as F

from invest.model.HGCN import RGCN
from invest.model.GRU import GRU


class SIPModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class SIP:
    def __init__(self):
        self.model = SIPModel()

    def fit(self):
        pass

    def predict(self):
        pass

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
