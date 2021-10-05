import numpy as np
import pandas as pd

from invest.utils import DataLoader, build_graph, load_data, dump_result, evaluate
from invest.model.GCN import GCN


train = load_data('data/movielens/movielens_train.csv')
train.columns = ['src_ind', 'dst_ind', 'label', 'date']
test = load_data('data/movielens/movielens_test.csv')
test.columns = ['src_ind', 'dst_ind', 'label', 'date']
n_nodes = np.concatenate([train['src_ind'], train['dst_ind'], test['src_ind'], test['dst_ind']]).max() + 1

train_loader = DataLoader(train, n_nodes, batch_size=50000)
g = build_graph(train)

model = GCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes, n_layers=2)
model.load('GCN-2.snapshot')

users, items, preds = [], [], []
item = list(train.dst_ind.unique())
for user in train.src_ind.unique():
    user = [user] * len(item) 
    users.extend(user)
    items.extend(item)
    preds.extend(list(model.predict(user, item, is_list=True)))

all_predictions = pd.DataFrame(data={"src_ind": users, "dst_ind":items, "prediction":preds})

merged = pd.merge(train, all_predictions, on=["src_ind", "dst_ind"], how="outer")
all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

metrics = evaluate(test, all_predictions)
print(metrics)
