# Hybrid model with RGCN for spectral recommendation and GRU for sequential recommendation
# SIP for spectral and sequential investment prediction

import pandas as pd
from datetime import datetime
import dgl
from loguru import logger

from invest.utils import build_hetero_graph, format_metrics, load_data, dump_result, evaluate, make_path
from invest.dataloader import CSNetSamplingDataLoader
from invest.model.CSNet import CSNet


path = 'data/tyc/'
save_path = f'ws/CSNet/{datetime.now().strftime("%Y%m%d%H%M%S")}/'
make_path(save_path)

logger.add(save_path + 'train.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
logger.info('******* CSNet *******')

# load raw data
logger.info('loading data')
train = load_data(path + 'train.csv')
valid = load_data(path + 'valid.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data(path + 'test.csv')
test_neg = load_data(path + 'test_neg.csv')

# load and build graphs
logger.info('load and build heterogeneous graph')
n_nodes = len(load_data(path + 'comps_total.csv'))
edge_dfs = {
    'label': train,          # train df first
    'gudong': load_data(path + 'comp_gudong_comp.csv'),
    'gongying': load_data(path + 'comp_gongying_comp.csv'),
    'dwtz': load_data(path + 'comp_dwtz_comp.csv'),
    # 'jingpin': load_data(path + 'comp_jingpin_comp.csv'),
    # 'lsgudong': load_data(path + 'comp_lsgudong_comp.csv'),
}
g = build_hetero_graph(edge_dfs, n_nodes, bidirectional=False)
big = build_hetero_graph(edge_dfs, n_nodes, bidirectional=True)
logger.info(g)

# build model
logger.info('building model')
model = CSNet(graph=g, in_feats=16, hid_feats=8, out_feats=4, n_nodes=n_nodes, rel_names=g.etypes)
logger.info(model.params)

# build sampler and data loader
block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_loader = CSNetSamplingDataLoader(train, g, big, block_sampler, neg_ratio=50, batch_size=50000)
test_loader = CSNetSamplingDataLoader(test, g, big, block_sampler, neg=test_neg, batch_size=100000)

# train and save model
print('training...')
model.fit(train_loader, test_loader, epoch=100, weight_decay=1e-2, lr=1e-2, print_every=1)
model.save(save_path + 'model.snapshot')

print('training finished')

# predict
# model.load('model/RGCN.snapshot')
print('predicting...')
pred = model.predict(test_loader)
dump_result(pred, save_path + f'result.csv')

# evaluate
metrics = evaluate(test, pred, top_k=[5, 10, 20])
logger.info(format_metrics(metrics))

# [mae: 0.2866] [rmse: 0.3065] [auc: 0.5987]
# [precision@5: 0.0933] [recall@5: 0.2799] [map@5: 0.1525] [ndcg@5: 0.2055]
# [precision@10: 0.0719] [recall@10: 0.3722] [map@10: 0.1699] [ndcg@10: 0.2415]
# [precision@20: 0.0480] [recall@20: 0.4284] [map@20: 0.1774] [ndcg@20: 0.2618]
