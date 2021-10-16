from dgl.batch import batch
import pandas as pd
import dgl
import os
import torch

from invest.metrics import auc, mae, precision_at_k, recall_at_k, ndcg_at_k, map_at_k, rmse


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


def evaluate(y, pred, top_k=5):
    metrics = {
        'mae': mae(y, pred),
        'rmse': rmse(y, pred),
        'auc': auc(y, pred)
    }
    if not isinstance(top_k, list):
        top_k = [top_k]
    for k in top_k:
        metrics[f'precision@{k}'] = precision_at_k(y, pred, k=k)
        metrics[f'recall@{k}'] = recall_at_k(y, pred, k=k)
        metrics[f'map@{k}'] = map_at_k(y, pred, k=k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(y, pred, k=k)
    return metrics


def format_metrics(metrics):
    return ' '.join(f'[{key}: {value:.4f}]' for key, value in metrics.items())


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


def build_hetero_graph(edge_dfs, n_nodes):
    graph_data = {}
    for key, df in edge_dfs.items():
        graph_data[('comp', key, 'comp')] = (df['src_ind'], df['dst_ind'])
    return dgl.heterograph(graph_data, num_nodes_dict={'comp': n_nodes})
