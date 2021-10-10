import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import pandas as pd
from loguru import logger
from tqdm import tqdm

from invest.utils import evaluate


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

    def forward(self, graph, src, dst):
        # inputs are features of nodes
        emb = self.embedding.weight
        h = self.conv1(graph, {'comp': emb})
        h = {k: torch.tanh(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: torch.tanh(v) for k, v in h.items()}
        return h['comp'][src], h['comp'][dst]


class RGCN:
    def __init__(self, graph, **kwargs):
        # 构造好模型
        self.graph = graph
        self.model = RGCNModel(**kwargs)
        self.params = kwargs

    def fit(self, train_loader, test, test_neg, epoch=50, lr=0.01, device='cpu'):
        model = self.model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
        for e in range(epoch):
            logger.info(f'【epoch {e + 1}】')
            it = tqdm(train_loader)
            self.model.train()
            for batch in it:
                src_feat, dst_feat = self.model(self.graph, torch.tensor(batch['src_ind']), torch.tensor(batch['dst_ind']))
                pred = torch.bmm(src_feat.unsqueeze(dim=1), dst_feat.unsqueeze(dim=2)).squeeze()
                loss = F.binary_cross_entropy_with_logits(pred, torch.tensor(batch['label']).clamp(0, 1).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                it.set_postfix({
                    'epoch': e + 1,
                    'loss': loss.item()
                })
        
            logger.info(f'epoch {e+1}: loss: {loss}')

            if (e + 1) % 2 != 0:
                    continue

            pred = self.predict(test)
            pred_neg = self.predict(test_neg)
            pred = pd.concat([pred, pred_neg], ignore_index=True)
            pred = pred.sample(frac=1).reset_index(drop=True)
            metrics = evaluate(test, pred, top_k=5)
            print(metrics)
            metrics = evaluate(test, pred, top_k=10)
            print(metrics)
            metrics = evaluate(test, pred, top_k=20)
            print(metrics)


    def predict(self, test):
        self.model.eval()
        src_feat, dst_feat = self.model(self.graph, torch.tensor(test['src_ind']), torch.tensor(test['dst_ind']))
        pred = torch.bmm(src_feat.unsqueeze(dim=1), dst_feat.unsqueeze(dim=2)).squeeze()
        test['prediction'] = pred.detach().numpy()
        return test

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

