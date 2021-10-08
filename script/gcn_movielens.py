import numpy as np
import pandas as pd

from invest.utils import DataLoader, build_graph, load_data, dump_result, evaluate
from invest.model.GCN import GCN


# load raw data
print('loading data')
train = load_data('data/movielens/movielens_train.csv')
train.columns = ['src_ind', 'dst_ind', 'label', 'date']

test = load_data('data/movielens/movielens_test.csv')
test.columns = ['src_ind', 'dst_ind', 'label', 'date']

n_users = np.concatenate([train['src_ind'], test['src_ind']]).max()
train['dst_ind'] += n_users
test['dst_ind'] += n_users
n_nodes = np.concatenate([train['dst_ind'], test['dst_ind']]).max() + 1

# build data loader
train_loader = DataLoader(train, n_nodes, batch_size=5000, neg_ratio=1)

# build graph
g = build_graph(train, n_nodes)

# build model
print('building model')
model = GCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes, n_layers=3)

# train and save model
print('training...')
model.fit(train_loader, test, test_neg=pd.DataFrame({'src_ind': [0], 'dst_ind': [0], 'label': [0]}), epoch=2)
model.save('GCN-2-movie.snapshot')

print('training finished')

# model.load('GCN-2-movie.snapshot')

print('predicting...')
users, items, preds = [], [], []
item = list(train.dst_ind.unique())
for user in train.src_ind.unique():
    user = [user] * len(item) 
    users.extend(user)
    items.extend(item)

all_predictions = model.predict(pd.DataFrame({'src_ind': users, 'dst_ind': items, 'rating': np.nan}))

merged = pd.merge(train, all_predictions, on=["src_ind", "dst_ind"], how="outer")
all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

metrics = evaluate(test, all_predictions)
print(metrics)
