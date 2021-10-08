import pandas as pd
import numpy as np
import random
import dgl

from torch.nn.functional import batch_norm
from invest.metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k


def load_data(path):
    return pd.read_csv(path)


def dump_result(df, path):
    df.to_csv(path, index=False)


def evaluate(y, pred, top_k=5):
    metrics = {
        f'precision@{top_k}': precision_at_k(y, pred, k=top_k),
        f'recall@{top_k}': recall_at_k(y, pred, k=top_k),
        f'ndcg@{top_k}': ndcg_at_k(y, pred, k=top_k),
        f'map@{top_k}': map_at_k(y, pred, k=top_k)
    }
    return metrics


class DataLoader:
    def __init__(self, data, n_nodes, neg=None, batch_size=32, neg_ratio=4, shuffle=True):
        self.data = data
        self.n_nodes = n_nodes
        self.neg = neg
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.shuffle = shuffle

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

        dst = set(self.data['dst_ind'])

        # interacted = self.data.groupby(self.data['src_ind']).apply(lambda x: set(x['dst_ind']))
        # neg_dst = [random.choice(list(dst - interacted[src])) for src in neg_src]
        # neg_dst = [random.choice(dst) for _ in range(len(neg_src))]
        neg_dst = np.random.randint(0, self.n_nodes, len(neg_src))
        neg_df = pd.DataFrame({'src_ind': neg_src, 'dst_ind': neg_dst, 'label': 0, 'date': np.nan})
        return neg_df


def build_graph(data, num_nodes):
    return dgl.graph((data['src_ind'], data['dst_ind']), num_nodes=num_nodes)


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
