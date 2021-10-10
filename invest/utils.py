from dgl.batch import batch
import pandas as pd
import numpy as np
import random
import dgl
import os
from datetime import datetime
import torch

from invest.metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k


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
        f'precision@{top_k}': precision_at_k(y, pred, k=top_k),
        f'recall@{top_k}': recall_at_k(y, pred, k=top_k),
        f'ndcg@{top_k}': ndcg_at_k(y, pred, k=top_k),
        f'map@{top_k}': map_at_k(y, pred, k=top_k)
    }
    return metrics


def print_metrics(metrics):
    print(', '.join(f'{key}: {value:.4f}' for key, value in metrics.items()))


class DataLoader:
    def __init__(self, data, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True):
        self.data = data
        self.n_nodes = n_nodes
        self.neg = neg
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.shuffle = shuffle
        self.params = {'batch_size': batch_size, 'neg_ratio': neg_ratio}

    def __len__(self):
        if self.neg:
            return (len(self.data) + len(self.neg)) // self.batch_size
        return len(self.data) * (1 + self.neg_ratio) // self.batch_size

    def __iter__(self):
        batch_size = self.batch_size

        if self.neg is None:
            data = pd.concat([self.data, self.neg_sample()], ignore_index=True)
        else:
            data = pd.concat([self.data, self.neg], ignore_index=True)
        
        # shuffle data
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        for i in range(len(data) // self.batch_size):
            yield data[i*batch_size:(i+1)*batch_size].reset_index(drop=True)

    def neg_sample(self):
        neg_src = np.repeat(self.data['src_ind'], self.neg_ratio)

        dst = list(set(self.data['dst_ind']))

        # interacted = self.data.groupby(self.data['src_ind']).apply(lambda x: set(x['dst_ind']))
        # neg_dst = [random.choice(list(dst - interacted[src])) for src in neg_src]
        neg_dst = [random.choice(dst) for _ in range(len(neg_src))]
        # neg_dst = np.random.randint(0, self.n_nodes, len(neg_src))
        neg_df = pd.DataFrame({'src_ind': neg_src, 'dst_ind': neg_dst, 'label': 0, 'date': np.nan})
        return neg_df


def build_graph(data, n_nodes):
    return dgl.graph((data['src_ind'], data['dst_ind']), num_nodes=n_nodes)


def build_multi_graph(edge_dfs, n_nodes):
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


def _create_bpr_loss(self, users, pos_items, neg_items):
    """Calculate BPR loss.

    Args:
        users (tf.Tensor): User embeddings to calculate loss.
        pos_items (tf.Tensor): Positive item embeddings to calculate loss.
        neg_items (tf.Tensor): Negative item embeddings to calculate loss.

    Returns:
        tf.Tensor, tf.Tensor: Matrix factorization loss. Embedding regularization loss.

    """
    pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
    neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

    regularizer = (
        tf.nn.l2_loss(self.u_g_embeddings_pre)
        + tf.nn.l2_loss(self.pos_i_g_embeddings_pre)
        + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
    )
    regularizer = regularizer / self.batch_size
    mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
    emb_loss = self.decay * regularizer
    return mf_loss, emb_loss
