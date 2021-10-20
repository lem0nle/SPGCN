import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import pandas as pd
from loguru import logger
from tqdm import tqdm

from invest.utils import evaluate, format_metrics


class RGCNModel(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_nodes, rel_names):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, in_feats)

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, g, input):
        # inputs are features of nodes
        emb = self.embedding(input)
        h = self.conv1(g[0], {'comp': emb})
        h = {k: torch.tanh(v) for k, v in h.items()}
        h = self.conv2(g[1], h)
        # h = {k: torch.tanh(v) for k, v in h.items()}
        return h['comp']


class RGCN:
    def __init__(self, graph, **kwargs):
        # 构造好模型
        self.graph = graph
        self.model = RGCNModel(**kwargs)
        self.params = kwargs

    def fit(self, train_loader, test_loader, epoch=50, lr=0.01, print_every=5, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
        for e in range(epoch):
            logger.info(f'【epoch {e + 1}】')
            it = tqdm(train_loader)
            self.model.train()
            for batch, (mfg, out_ind) in it:
                pred = self.predict_batch(mfg, out_ind)
                loss = F.binary_cross_entropy_with_logits(pred, torch.tensor(batch['label']).clamp(0, 1).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                it.set_postfix({
                    'epoch': e + 1,
                    'loss': loss.item()
                })
        
            logger.info(f'epoch {e+1}: loss: {loss}')

            if (e + 1) % print_every != 0:
                    continue

            pred = self.predict(test_loader)
            metrics = evaluate(test_loader.data, pred, top_k=[5, 10, 20])
            logger.info(format_metrics(metrics))

    def predict_batch(self, mfg, out_ind):
        input = mfg[0].srcdata['_ID']
        emb = self.model(mfg, input)
        output = emb[out_ind]
        batch_len = len(output) // 2
        src_feat, dst_feat = output[:batch_len], output[batch_len:]
        batch_pred = torch.bmm(src_feat.unsqueeze(dim=1), dst_feat.unsqueeze(dim=2)).squeeze()
        return batch_pred

    def predict(self, test_loader):
        self.model.eval()
        it = tqdm(test_loader)
        src_inds = []
        dst_inds = []
        preds = []
        with torch.no_grad():
            for batch, (mfg, out_ind) in it:
                pred = self.predict_batch(mfg, out_ind)
                src_inds.append(batch['src_ind'])
                dst_inds.append(batch['dst_ind'])
                preds.append(pred.numpy())

        df = pd.DataFrame({
            'src_ind': np.concatenate(src_inds),
            'dst_ind': np.concatenate(dst_inds),
            'prediction': np.concatenate(preds),
        })
        return df

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

