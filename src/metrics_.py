import pandas as pd
from sklearn.metrics import mean_absolute_error, ndcg_score, roc_auc_score, average_precision_score, top_k_accuracy_score


if __name__ == '__main__':
    df = pd.read_csv('data/result_tmp.csv')
    # MAE
    # mae = df.groupby('src_ind').apply(lambda x: (x.label-x.score).abs().mean()).mean()
    mae = df.groupby('src_ind').apply(lambda x: mean_absolute_error(x.label, x.score)).mean()

    # AUC
    auc = df.groupby('src_ind').apply(lambda x: roc_auc_score(x.label, x.score)).mean()

    # NDCG@5
    ndcg = df.groupby('src_ind').apply(lambda x: ndcg_score([x.label], [x.score], k=5)).mean()

    # HR@1
    hr = df.groupby('src_ind').apply(lambda x: top_k_accuracy_score(x.label, x.score, k=1)).mean()

    print(mae, auc, ndcg, hr)
