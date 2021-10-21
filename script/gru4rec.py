from typing import Counter
import pandas as pd
from loguru import logger
from invest.utils import Acc, evaluate, format_metrics, load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

path = 'data/tyc/'

# load raw data
logger.info('loading data')
train = load_data(path + 'train.csv')
valid = load_data(path + 'valid.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data(path + 'test.csv')
test_neg = load_data(path + 'test_neg.csv')

n_nodes = np.concatenate([train['src_ind'], train['dst_ind'], test['src_ind'], test['dst_ind']]).max() + 1

test_all = pd.concat([test, test_neg], ignore_index=True)

train_seqs = train.groupby('src_ind')['dst_ind'].apply(list)
test_seqs = test_all.groupby('src_ind')['dst_ind'].apply(list)
test_labels = test_all.groupby('src_ind')['label'].apply(list)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, 32)
        self.rnn = nn.GRU(32, 32, num_layers=1)
        self.output = nn.Linear(32, n_nodes)
    def forward(self, x):
        y, _ = self.rnn(self.emb(x).unsqueeze(1))
        return self.output(y).squeeze(1)

model = Model()


def train(print_every=5):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(100):
        logger.info('epoch: {}', e + 1)
        model.train()
        epoch_loss = Acc()
        it = tqdm(train_seqs.iteritems(), total=len(train_seqs))
        for _, seq in it:
            if len(seq) < 2:
                continue
            seq = seq.copy()
            random.shuffle(seq)
            x = torch.tensor(seq)
            loss = F.cross_entropy(model(x[:-1]), x[1:])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            it.set_postfix({
                'loss': epoch_loss.mean()
            })
        
        if (e + 1) % print_every != 0:
            continue

        predict()


def predict():
    model.eval()
    with torch.no_grad():
        it = tqdm(test_seqs.iteritems(), total=len(test_seqs))
        src_ind = []
        dst_ind = []
        prediction = []
        for user, seq in it:
            try:
                train_seq = train_seqs.loc[user]
            except KeyError:
                continue
            x = torch.tensor(train_seq)
            y = model(x)
            pred = y[-1][seq]
            src_ind.extend([user] * len(pred))
            dst_ind.extend(seq)
            prediction.extend(pred.tolist())
        pred = pd.DataFrame({
            'src_ind': src_ind,
            'dst_ind': dst_ind,
            'prediction': prediction,
        })
        metrics = evaluate(test, pred, [5, 10, 20])
        print(format_metrics(metrics))


predict()
train()


