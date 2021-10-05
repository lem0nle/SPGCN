import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
# import numpy as np
import pandas as pd
from torch.nn.modules.sparse import Embedding
from tqdm import tqdm
import logging

from invest.utils import evaluate


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('GCN')
# logger.setLevel(logging.INFO)


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = torch.tanh

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        h = g.ndata.pop('h')
        h = self.linear(h)
        h = self.activation(h)
        return h


class GCNModel(nn.Module):
    def __init__(self, in_feats, out_feats, n_nodes, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, in_feats)
        modules = [
            GCNLayer(in_feats, out_feats),
        ]
        modules.extend([GCNLayer(out_feats, out_feats)
                        for _ in range(n_layers - 1)])
        self.module_list = nn.ModuleList(modules)
    
    def forward(self, g, src, dst):
        emb = self.embedding.weight
        for layer in self.module_list:
            emb = layer(g, emb)
        return emb[src], emb[dst]


class GCN:
    def __init__(self, graph, **kwargs):
        # 构造好模型
        self.graph = graph
        self.model = GCNModel(**kwargs)

    def fit(self, train_loader, test, test_neg, epoch=50, lr=0.01, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for e in range(epoch):
            logger.info(f'【epoch {e + 1}】')
            it = tqdm(train_loader)
            for batch in it:
                src_feat, dst_feat = self.model(self.graph, torch.tensor(batch['src_ind']), torch.tensor(batch['dst_ind']))
                pred = torch.bmm(src_feat.unsqueeze(dim=1), dst_feat.unsqueeze(dim=2)).squeeze()
                loss = F.binary_cross_entropy_with_logits(pred, torch.tensor(batch['label']).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                it.set_postfix({
                    'epoch': e + 1,
                    'loss': loss.item()
                })
        
            logger.info(f'epoch {e+1}: loss: {loss}')

            # if (e + 1) % 10 != 0:
            #         continue

            pred = self.predict(test)
            pred_neg = self.predict(test_neg)
            pred = pd.concat([pred, pred_neg], ignore_index=True)
            metrics = evaluate(test, pred)
            print(metrics)


    def predict(self, test):
        src_feat, dst_feat = self.model(self.graph, torch.tensor(test['src_ind']), torch.tensor(test['dst_ind']))
        pred = torch.bmm(src_feat.unsqueeze(dim=1), dst_feat.unsqueeze(dim=2)).squeeze()
        test['prediction'] = pred.detach().numpy()
        return test

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))