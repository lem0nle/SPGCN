# Hybrid model with RGCN for spectral recommendation and GRU for sequential recommendation
# SIP for spectral and sequential investment prediction

import pandas as pd
from datetime import datetime
import dgl
from loguru import logger

from invest.utils import build_hetero_graph, format_metrics, load_data, dump_result, evaluate, make_path
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
    # 'jingpin': load_data(path + 'comp_jingpin_comp.csv'),
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
train_loader = BlockSeqSamplingDataLoader(train, g, block_sampler, random_walk_sampler, batch_size=512)
test_loader = BlockSeqSamplingDataLoader(test, g, block_sampler, random_walk_sampler, neg=test_neg, batch_size=2048)

# train and save model
print('training...')
model.fit(train_loader, test_loader, epoch=20, print_every=1)
model.save(save_path + 'SIPmodel.snapshot')

print('training finished')

# predict
# model.load('model/RGCN.snapshot')
print('predicting...')
pred = model.predict(test_loader)
dump_result(pred, save_path + f'result.csv')

# evaluate
metrics = evaluate(test, pred, top_k=[5, 10, 20])
logger.info(format_metrics(metrics))

# best
# {'precision@5': 0.09474358974358973, 'recall@5': 0.2827203505248301, 'ndcg@5': 0.21005569685684367, 'map@5': 0.15752924419076567}
# {'precision@10': 0.06903846153846155, 'recall@10': 0.35773097092608824, 'ndcg@10': 0.23901717902527395, 'map@10': 0.17197623824856345}
# {'precision@20': 0.04653846153846154, 'recall@20': 0.42097959158274495, 'ndcg@20': 0.2604866439551872, 'map@20': 0.1793107046030604}
