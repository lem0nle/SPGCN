import numpy as np
import pandas as pd

from invest.utils import DataLoader, build_graph, load_data, dump_result, evaluate
from invest.model.GCN import GCN


# load raw data
print('loading data')
train = load_data('data/train_df_.csv')
valid = load_data('data/valid_df_.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data('data/test_df_.csv')
test_neg = load_data('data/test_neg_df_.csv')
n_nodes = np.concatenate([train['src_ind'], train['dst_ind'], test['src_ind'], test['dst_ind']]).max() + 1

# # build data loader
train_loader = DataLoader(train, n_nodes, batch_size=10000)
# # test_loader = DataLoader(test, n_nodes, batch_size=256, neg=test_neg, shuffle=False)

# build graph
g = build_graph(train, n_nodes)
# g = build_graph(pd.concat([train, test], ignore_index=True), n_nodes)

# build model
print('building model')
model = GCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes)

# # train and save model
print('training...')
model.fit(train_loader, test, test_neg, epoch=200)
model.save('LightGCN.snapshot')

print('training finished')

# predict
model.load('LightGCN.snapshot')
print('predicting...')
pred = model.predict(test)
pred_neg = model.predict(test_neg)
pred = pd.concat([pred, pred_neg], ignore_index=True)
pred = pred.sample(frac=1).reset_index(drop=True)
dump_result(pred, 'result/gcn/gcn_20.csv')

# evaluate
metrics = evaluate(test, pred, top_k=5)
print(metrics)

# best:
# {'precision@5': 0.011764705882352938, 'recall@5': 0.03613709020605105, 'ndcg@5': 0.025209521323411817, 'map@5': 0.017221131468785882}
# {'precision@10': 0.012790697674418601, 'recall@10': 0.07629542419686673, 'ndcg@10': 0.03909899744081252, 'map@10': 0.022542016817142775}
# {'precision@20': 0.013337893296853625, 'recall@20': 0.15492285064355768, 'ndcg@20': 0.061127759186525445, 'map@20': 0.02811449181881521}