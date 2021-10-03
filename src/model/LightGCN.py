import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LightGCN(nn.Module):
    def __init__(self, n_layer=3):
        self.n_layer = n_layer

    def fit(self):
        pass

    def predict(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
