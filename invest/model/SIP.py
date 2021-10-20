from unicodedata import bidirectional
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from invest.model.HGCN import RGCNModel
from invest.utils import Acc, batch_dot, evaluate, format_metrics


class SIPModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class SIP:
    def __init__(self, graph, **kwargs):
        self.params = kwargs
        # 构造好模型
        self.graph = graph
        self.model = RGCNModel(**kwargs)
        self.label_model = RGCNModel(**kwargs)
        self.frnn = nn.GRU(kwargs['in_feats'], kwargs['out_feats'], num_layers=1)
        self.brnn = nn.GRU(kwargs['in_feats'], kwargs['out_feats'], num_layers=1)

    def fit(self, train_loader, test_loader, epoch=50, lr=0.01, weight_decay=4e-3, print_every=5, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.AdamW(
            chain(model.parameters(), self.label_model.parameters(), self.frnn.parameters(), self.brnn.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        for e in range(epoch):
            logger.info(f'【epoch {e + 1}】')
            it = tqdm(train_loader)
            self.model.train()
            self.label_model.train()
            self.frnn.train()
            self.brnn.train()
            batch_loss = Acc()
            for batch, (mfg, out_ind), (mfg_, out_ind_), seqs in it:
                pred = self.predict_batch(mfg, out_ind, mfg_, out_ind_, seqs)
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

    def predict_batch(self, mfg, out_ind, mfg_, out_ind_, seqs):
        # spatial
        input = mfg[0].srcdata['_ID']
        emb = self.model(mfg, input)
        output = emb[out_ind]
        batch_len = len(output) // 2
        src_feat_g, dst_feat_g = output[:batch_len], output[batch_len:]

        # contextual
        seq_len = seqs.size(0)
        embs = self.model.embedding(seqs).detach()
        src_emb, dst_emb = embs[seq_len // 2, :batch_len], embs[seq_len // 2, batch_len:]
        _, h_f = self.frnn(embs[:seq_len // 2])
        _, h_b = self.brnn(embs[(seq_len + 1) // 2:].flip(0))
        h = h_f + h_b
        src_feat_s, dst_feat_s = h[0, :batch_len], h[0, batch_len:]

        # collaborate filtering
        input = mfg_[0].srcdata['_ID']
        emb = self.model(mfg_, input)
        output = emb[out_ind_]
        batch_len = len(output) // 2
        src_feat_c, dst_feat_c = output[:batch_len], output[batch_len:]

        # merge
        a = torch.softmax(torch.cat([batch_dot(src_feat_c, dst_feat_g), batch_dot(src_feat_c, dst_feat_s), batch_dot(src_feat_c, dst_feat_c)], dim=-1), dim=-1)
        dst_feat_merge = a[:, :1] * dst_feat_g + a[:, 1:2] * dst_feat_s + a[:, 2:] * dst_feat_c
        dst_feat = dst_feat_merge # + dst_feat_c

        # predict
        batch_pred = batch_dot(src_feat_c, dst_feat).squeeze()

        return batch_pred

    def predict(self, test_loader):
        self.model.eval()
        self.label_model.eval()
        self.frnn.eval()
        self.brnn.eval()
        it = tqdm(test_loader)
        src_inds = []
        dst_inds = []
        preds = []
        with torch.no_grad():
            for batch, (mfg, out_ind), (mfg_, out_ind_), seqs in it:
                pred = self.predict_batch(mfg, out_ind, mfg_, out_ind_, seqs)
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
