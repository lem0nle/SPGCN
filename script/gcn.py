import numpy as np
import pandas as pd

from invest.utils import DataLoader, build_graph, load_data, dump_result, evaluate
from invest.model.GCN import GCN


# load raw data
print('loading data')
train = load_data('data/movielens/movielens_train.csv')
train.columns = ['src_ind', 'dst_ind', 'label', 'date']

# train = load_data('data/train_df_.csv')
test = load_data('data/movielens/movielens_test.csv')
test.columns = ['src_ind', 'dst_ind', 'label', 'date']

# test = load_data('data/test_df_.csv')
# test_neg = load_data('data/test_neg_df_user_.csv')
n_nodes = np.concatenate([train['src_ind'], train['dst_ind'], test['src_ind'], test['dst_ind']]).max() + 1

# build data loader
train_loader = DataLoader(train, n_nodes, batch_size=50000)
# test_loader = DataLoader(test, n_nodes, batch_size=256, neg=test_neg, shuffle=False)

# build graph
g = build_graph(train, n_nodes)

# build model
print('building model')
model = GCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes, n_layers=2)

# train and save model
print('training...')
model.fit(train_loader, test, test_neg=pd.DataFrame({'src_ind': [], 'dst_ind': [], 'label': []}), epoch=20)
model.save('GCN-2-movie.snapshot')

print('training finished')

# predict
# print('predicting...')
# pred = model.predict(test)
# pred_neg = model.predict(test_neg)
# pred = pd.concat([pred, pred_neg], ignore_index=True)
# dump_result(pred, 'result/gcn/gcn-2_20.csv')

# # evaluate
# metrics = evaluate(test, pred)
# print(metrics)
