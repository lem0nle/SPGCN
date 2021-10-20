import numpy as np
import pandas as pd
import dgl
import os
import torch

from invest.metrics import auc, precision_at_k, recall_at_k, ndcg_at_k, map_at_k


def load_data(path, before_date=None):
    df = pd.read_csv(path)

    if before_date is not None:
        df = df[df.date < before_date]
    
    return df


def make_path(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def dump_result(df, path):
    make_path(path)
    df.to_csv(path, index=False)


def evaluate(y, pred, top_k=5, sigmoid=True):
    if sigmoid:
        pred['prediction_'] = (np.tanh(pred['prediction']) + 1) / 2
    metrics = {
        # mae and rmse are too sensitive to hyper-parameter change
        # 'mae': mae(y, pred, col_prediction='prediction_'),
        # 'rmse': rmse(y, pred, col_prediction='prediction_'),
        'auc': auc(y, pred, col_prediction='prediction_' if sigmoid else 'prediction')
    }
    if not isinstance(top_k, list):
        top_k = [top_k]
    for k in top_k:
        metrics[f'precision@{k:02}'] = precision_at_k(y, pred, k=k)
        metrics[f'recall@{k:02}'] = recall_at_k(y, pred, k=k)
        metrics[f'map@{k:02}'] = map_at_k(y, pred, k=k)
        metrics[f'ndcg@{k:02}'] = ndcg_at_k(y, pred, k=k)
    return metrics


def format_metrics(metrics):
    return ' '.join(sorted(f'[{key}: {value:.4f}]' for key, value in metrics.items()))


def build_graph(data, n_nodes):
    return dgl.graph((data['src_ind'], data['dst_ind']), num_nodes=n_nodes)


def build_multi_graph(edge_dfs, n_nodes):
    edge_dfs = list(edge_dfs.values())
    n_rels = len(edge_dfs)
    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    node_rel_cnt = torch.zeros(n_nodes, n_rels).float()

    for r in range(len(edge_dfs)):
        df = edge_dfs[r][['src_ind', 'dst_ind']]
        g.add_edges(df['src_ind'], df['dst_ind'], {'rel_type': torch.tensor([[r]] * len(df))})
        for _, row in df.iterrows():
            node_rel_cnt[row['dst_ind'], r] += 1

    edge_norm = []
    for _, v, r in zip(*g.all_edges(), g.edata['rel_type']):
        norm = 1. / (node_rel_cnt[v, r] + 1e-9)
        edge_norm.append(norm)

    g.edata['norm'] = torch.stack(edge_norm)

    return g


def build_hetero_graph(edge_dfs, n_nodes, bidirectional=False):
    graph_data = {}
    for key, df in edge_dfs.items():
        graph_data[('comp', key, 'comp')] = (df['src_ind'], df['dst_ind'])
        if bidirectional:
            graph_data[('comp', key + '_inv', 'comp')] = (df['dst_ind'], df['src_ind'])
    return dgl.heterograph(graph_data, num_nodes_dict={'comp': n_nodes})


def batch_dot(x, y):
    return torch.bmm(x.unsqueeze(dim=1), y.unsqueeze(dim=2)).squeeze(-1)


class Acc:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def __iadd__(self, other):
        self.sum += other
        self.count += 1
        return self
    def mean(self):
        return self.sum / self.count
