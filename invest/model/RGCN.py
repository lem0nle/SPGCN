import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pandas as pd
from loguru import logger
from tqdm import tqdm

from invest.utils import evaluate, format_metrics


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.activation = torch.tanh

        self.weight = nn.Parameter(torch.empty(self.num_rels,
                                               self.in_feat,
                                               self.out_feat))
        self.bias = nn.Parameter(torch.empty(1, out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, g, feature):
        with g.local_scope():
            g.srcdata['h'] = feature
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'),
                        self.apply_func)
            return g.dstdata.pop('h')

    def message_func(self, edges):
        weight = self.weight
        w = weight[edges.data['rel_type'].squeeze()]

        h = edges.src['h']
        sizes = h.size()
        h = h.view(w.size(0), -1, sizes[-1])
        msg = torch.bmm(h, w)
        msg *= edges.data['norm'].unsqueeze(1)
        msg = msg.view(*sizes[:-1], -1)
        return {'msg': msg}

    def apply_func(self, nodes):
        h = nodes.data['h']
        bias = self.bias
        for _ in range(h.dim() - bias.dim()):
            bias = bias[None, ...]
        h = h + bias
        h = self.activation(h)
        return {'h': h}


class RGCNModel(nn.Module):
    def __init__(self, in_feats, out_feats, n_nodes, n_layers=3, num_rels=2):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, in_feats)

        modules = [
            RGCNLayer(in_feats, out_feats, num_rels),
        ]
        modules.extend([RGCNLayer(out_feats, out_feats, num_rels)
                        for _ in range(n_layers - 1)])
        self.module_list = nn.ModuleList(modules)

    def forward(self, g, input):
        emb = self.embedding(input)
        for i, layer in enumerate(self.module_list):
            emb = layer(g[i], emb)
        return emb


class RGCN:
    def __init__(self, graph, **kwargs):
        # 构造好模型
        self.graph = graph
        self.model = RGCNModel(**kwargs)

    def fit(self, train_loader, test_loader, epoch=50, lr=0.01, print_every=5, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
        
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
