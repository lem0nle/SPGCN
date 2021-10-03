import pandas as pd
from src.metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k


def load_data(path):
    return pd.read_csv(path)


def dump_result(df, path):
    df.to_csv(path, index=False)


def evaluate(y, pred, top_k=5):
    metrics = {
        'precision@k': precision_at_k(y, pred, top_k),
        'recall@k': recall_at_k(y, pred, top_k),
        'ndcg@k': ndcg_at_k(y, pred, top_k),
        'map@k': map_at_k(y, pred, top_k)
    }
    return metrics
