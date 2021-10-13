# Hybrid model with RGCN for spectral recommendation and GRU for sequential recommendation
# SIP for spectral and sequential investment prediction

import pandas as pd
from datetime import datetime
import dgl
from loguru import logger

from invest.utils import build_hetero_graph, load_data, dump_result, evaluate, make_path, print_metrics
from invest.dataloader import BlockSeqSamplingDataLoader, RandomWalkSampler
from invest.model.SIP import SIP


path = 'data/tyc/'
save_path = f'ws/SIP/{datetime.now().strftime("%Y%m%d%H%M%S")}/'
make_path(save_path)

logger.add(save_path + 'train.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
logger.info('******* SIP *******')

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
    'jingpin': load_data(path + 'comp_jingpin_comp.csv'),
    'lsgudong': load_data(path + 'comp_lsgudong_comp.csv'),
}
g = build_hetero_graph(edge_dfs, n_nodes)
logger.info(g)

# build model
logger.info('building model')
# model = SIP(graph=g, in_feats=64, hid_feats=64, out_feats=64, input_size=64, hidden_size=64, n_nodes=n_nodes, rel_names=edge_dfs.keys())
model = SIP(graph=g, in_feats=64, hid_feats=64, out_feats=64, n_nodes=n_nodes, rel_names=edge_dfs.keys())
logger.info(model.params)

# build sampler and data loader
random_walk_sampler = RandomWalkSampler(5)
block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_loader = BlockSeqSamplingDataLoader(train, g, block_sampler, random_walk_sampler, batch_size=128)
test_loader = BlockSeqSamplingDataLoader(test, g, block_sampler, random_walk_sampler, neg=test_neg, batch_size=256)

# train and save model
print('training...')
model.fit(train_loader, test_loader, epoch=20)
model.save(save_path + 'SIPmodel.snapshot')

print('training finished')

# predict
# model.load('model/RGCN.snapshot')
print('predicting...')
pred = model.predict(test_loader)
dump_result(pred, save_path + f'result.csv')

# evaluate
metrics = evaluate(test, pred, top_k=5)
print_metrics(metrics)
logger.info(metrics)
metrics = evaluate(test, pred, top_k=10)
print_metrics(metrics)
logger.info(metrics)
metrics = evaluate(test, pred, top_k=20)
print_metrics(metrics)
logger.info(metrics)
