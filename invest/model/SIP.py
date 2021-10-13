from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from invest.model.HGCN import RGCNModel
from invest.model.GRU import GRU
from invest.utils import evaluate


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

    def fit(self, train_loader, test_loader, epoch=50, lr=0.01, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
        
        for e in range(epoch):
            logger.info(f'【epoch {e + 1}】')
            it = tqdm(train_loader)
            self.model.train()
            for batch, (mfg, out_ind), seqs in it:
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

            if (e + 1) % 1 != 0:
                    continue

            pred = self.predict(test_loader)
            metrics = evaluate(test_loader.data, pred, top_k=5)
            print(metrics)
            metrics = evaluate(test_loader.data, pred, top_k=10)
            print(metrics)
            metrics = evaluate(test_loader.data, pred, top_k=20)
            print(metrics)

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
