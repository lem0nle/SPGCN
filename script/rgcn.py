# RGCN with relational graph feature for recommendation

import numpy as np
import pandas as pd

from invest.utils import DataLoader, build_hetero_graph, build_multi_graph, load_data, dump_result, evaluate
from invest.model.RGCN import RGCN

path = 'data/tyc/'

# load raw data
print('loading data')
train = load_data(path + 'train.csv')
valid = load_data(path + 'valid.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data(path + 'test.csv')
test_neg = load_data(path + 'test_neg.csv')

# load and build graphs
print('load and build multi-graph')
n_nodes = len(load_data(path + 'comps_total.csv'))
edge_dfs = {
    'label': train,          # train df first
    'gudong': load_data(path + 'comp_gudong_comp.csv'),
    'gongying': load_data(path + 'comp_gongying_comp.csv'),
    'dwtz': load_data(path + 'comp_dwtz_comp.csv'),
    'jingpin': load_data(path + 'comp_jingpin_comp.csv'),
    'lsgudong': load_data(path + 'comp_lsgudong_comp.csv'),
}
g = build_multi_graph(edge_dfs, n_nodes)

# build model
print('building model')
model = RGCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes, num_rels=len(edge_dfs))

# build data loader
train_loader = DataLoader(train, batch_size=10000)

# train and save model
print('training...')
model.fit(train_loader, test, test_neg, epoch=20)
model.save('model/RGCN.snapshot')

print('training finished')

# predict
# model.load('model/RGCN.snapshot')
print('predicting...')
pred = model.predict(test)
pred_neg = model.predict(test_neg)
pred = pd.concat([pred, pred_neg], ignore_index=True)
pred = pred.sample(frac=1).reset_index(drop=True)
dump_result(pred, 'result/rgcn/rgcn_20.csv')

# evaluate
metrics = evaluate(test, pred, top_k=5)
print(metrics)
