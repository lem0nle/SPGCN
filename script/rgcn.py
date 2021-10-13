# RGCN with relational graph feature for recommendation

import dgl
import pandas as pd
from datetime import datetime
from loguru import logger

from invest.utils import build_multi_graph, load_data, dump_result, evaluate, make_path
from invest.dataloader import BlockSamplingDataLoader
from invest.model.RGCN import RGCN

path = 'data/tyc/'
save_path = f'ws/RGCN/{datetime.now().strftime("%Y%m%d%H%M%S")}/'
make_path(save_path)

logger.add(save_path + 'train.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
logger.info('******* HGCN *******')

# load raw data
logger.info('loading data')
train = load_data(path + 'train.csv')
valid = load_data(path + 'valid.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data(path + 'test.csv')
test_neg = load_data(path + 'test_neg.csv')

# load and build graphs
logger.info('load and build multi-graph')
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
logger.info('building model')
model = RGCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes, num_rels=len(edge_dfs), n_layers=2)

# build data loader
block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_loader = BlockSamplingDataLoader(train, g, block_sampler, batch_size=128)
test_loader = BlockSamplingDataLoader(test, g, block_sampler, neg=test_neg, batch_size=256)

# train and save model
logger.info('training...')
model.fit(train_loader, test_loader, epoch=20)
model.save('model/RGCN.snapshot')

logger.info('training finished')

# predict
# model.load('model/RGCN.snapshot')
logger.info('predicting...')
pred = model.predict(test_loader)
dump_result(pred, 'result/rgcn/rgcn_20.csv')

# evaluate
metrics = evaluate(test, pred, top_k=5)
logger.info(metrics)

# best
# {'precision@5': 0.09333333333333332, 'recall@5': 0.2839836533161315, 'ndcg@5': 0.208160765796159, 'map@5': 0.15555669210048936}
# {'precision@10': 0.07301282051282053, 'recall@10': 0.3839026439467182, 'ndcg@10': 0.2470978743535754, 'map@10': 0.17432767068478447}
# {'precision@20': 0.050032051282051286, 'recall@20': 0.46823957665081845, 'ndcg@20': 0.27425916275994733, 'map@20': 0.18341113415598392}
