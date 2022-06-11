import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pandas as pd
from loguru import logger
from tqdm import tqdm

from invest.utils import Acc, batch_dot, evaluate, format_metrics


class CSNetModel(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_nodes, rel_names):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, in_feats)

        self.gcn1 = dglnn.GraphConv(in_feats, hid_feats, activation=torch.tanh, norm='none')
        self.gcn2 = dglnn.GraphConv(hid_feats, out_feats, activation=torch.tanh, norm='none')

        self.rgcn1 = dglnn.RelGraphConv(in_feats, hid_feats, len(rel_names) * 2, activation=torch.tanh, dropout=0.5)
        self.rgcn2 = dglnn.RelGraphConv(hid_feats, out_feats, len(rel_names) * 2, activation=torch.tanh, dropout=0.5)

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats, activation=torch.tanh, norm='none')
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats, activation=torch.tanh, norm='none')
            for rel in rel_names}, aggregate='mean')

    def forward(self, mfg, bimfg, lmfg):
        # inputs are features of nodes
        emb = self.embedding(lmfg[0].srcdata['_ID']).detach()
        h = self.gcn1(lmfg[0], emb)
        h = self.gcn2(lmfg[1], h)
        hcf = h

        emb = self.embedding(bimfg[0].srcdata['_ID']).detach()
        h = self.rgcn1(bimfg[0], emb, etypes=bimfg[0].edata[dgl.ETYPE])
        h = self.rgcn2(bimfg[1], h, etypes=bimfg[1].edata[dgl.ETYPE])
        hc = h

        emb = self.embedding(mfg[0].srcdata['_ID']).detach()
        h = self.conv1(mfg[0], {'comp': emb})
        h = self.conv2(mfg[1], h)
        hs = h['comp']

        return hcf, hc, hs


class CSNet:
    def __init__(self, graph, **kwargs):
        # 构造好模型
        self.graph = graph
        self.model = CSNetModel(**kwargs)
        self.params = kwargs

    def fit(self, train_loader, test_loader, epoch=50, lr=0.01, weight_decay=1e-3, print_every=5, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for e in range(epoch):
            logger.info(f'【epoch {e + 1}】')
            it = tqdm(train_loader)
            self.model.train()
            batch_loss = Acc()
            for batch, mfg, bimfg, lmfg, out_ind in it:
                pred = self.predict_batch(mfg, bimfg, lmfg, out_ind)
                loss = F.binary_cross_entropy_with_logits(pred, torch.tensor(batch['label']).clamp(0, 1).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

                it.set_postfix({
                    'epoch': e + 1,
                    'loss': batch_loss.mean()
                })
        
            logger.info(f'epoch {e+1}: loss: {batch_loss.mean()}')

            if (e + 1) % print_every != 0:
                    continue

            pred = self.predict(test_loader)
            metrics = evaluate(test_loader.data, pred, top_k=[5, 10, 20])
            logger.info(format_metrics(metrics))

    def _pred(self, mfg, bimfg, lmfg, out_ind):
        hcf, hc, hs = self.model(mfg, bimfg, lmfg)
        hcf = hcf[out_ind]
        hc = hc[out_ind]
        hs = hs[out_ind]
        return hcf, hc, hs

    def predict_batch(self, mfg, bimfg, lmfg, out_ind):
        hcf, hc, hs =  self._pred(mfg, bimfg, lmfg, out_ind)
        batch_len = len(hcf) // 2
        src_feat, dst_feat = hcf[:batch_len], hcf[batch_len:]
        _, dst_feat_c = hc[:batch_len], hc[batch_len:]
        _, dst_feat_s = hs[:batch_len], hs[batch_len:]
            

        a = torch.softmax(torch.cat([batch_dot(src_feat, dst_feat_c), batch_dot(src_feat, dst_feat_s), batch_dot(src_feat, dst_feat)], dim=-1), dim=-1)
        dst_feat_merge = a[:, :1] * dst_feat_c + a[:, 1:2] * dst_feat_s + a[:, 2:] * dst_feat

        # a = torch.softmax(torch.cat([batch_dot(src_feat, dst_feat_c), batch_dot(src_feat, dst_feat_s)], dim=-1), dim=-1)
        # dst_feat_merge = a[:, :1] * dst_feat_c + a[:, 1:] * dst_feat_s

        # a = torch.softmax(torch.cat([batch_dot(src_feat, dst_feat_c), batch_dot(src_feat, dst_feat)], dim=-1), dim=-1)
        # dst_feat_merge = a[:, :1] * dst_feat_c + a[:, 1:] * dst_feat

        # a = torch.softmax(torch.cat([batch_dot(src_feat, dst_feat_s), batch_dot(src_feat, dst_feat)], dim=-1), dim=-1)
        # dst_feat_merge = a[:, :1] * dst_feat_s + a[:, 1:] * dst_feat

        # dst_feat_merge = (dst_feat_c + dst_feat_s + dst_feat) / 3

        batch_pred = batch_dot(src_feat, dst_feat_merge).squeeze()
        return batch_pred

    def predict(self, test_loader):
        self.model.eval()
        it = tqdm(test_loader)
        src_inds = []
        dst_inds = []
        preds = []
        with torch.no_grad():
            for batch, mfg, bimfg, lmfg, out_ind in it:
                pred = self.predict_batch(mfg, bimfg, lmfg, out_ind)
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
