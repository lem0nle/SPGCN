# RGCN with heterogeneous graph feature for recommendation

import dgl
import pandas as pd
from datetime import datetime
from loguru import logger

from invest.utils import build_hetero_graph, format_metrics, load_data, dump_result, evaluate, make_path
from invest.dataloader import BlockSamplingDataLoader, DataLoader
from invest.model.HGCN import RGCN

path = 'data/tyc/'
save_path = f'ws/HGCN/{datetime.now().strftime("%Y%m%d%H%M%S")}/'
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

# build model
logger.info('building model')
model = RGCN(graph=g, in_feats=64, hid_feats=64, out_feats=64, n_nodes=n_nodes, rel_names=edge_dfs.keys())
logger.info(model.params)

# build data loader
block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_loader = BlockSamplingDataLoader(train, g, block_sampler, batch_size=4096)
test_loader = BlockSamplingDataLoader(test, g, block_sampler, neg=test_neg, batch_size=8192)


# train and save model
logger.info('training...')
params = {'epoch': 20, 'lr': 1e-2}
model.fit(train_loader, test_loader, epoch=params['epoch'], lr=params['lr'], print_every=1)
model.save(save_path + f'model_{params["epoch"]}_{params["lr"]}.snapshot')

logger.info('training finished')

# predict
# model.load(save_path + f"model_{params['epoch']}_{params['lr']}.snapshot")
logger.info('predicting...')
pred = model.predict(test_loader)
dump_result(pred, save_path + f'result_{params["epoch"]}_{params["lr"]}.csv')

# evaluate
metrics = evaluate(test, pred, top_k=[5, 10, 20])
logger.info(format_metrics(metrics))

# [mae: 0.3680] [rmse: 0.3951] [auc: 0.5909]
# [precision@5: 0.0685] [recall@5: 0.2436] [map@5: 0.1042] [ndcg@5: 0.1473]
# [precision@10: 0.0576] [recall@10: 0.3428] [map@10: 0.1213] [ndcg@10: 0.1879]
# [precision@20: 0.0430] [recall@20: 0.4209] [map@20: 0.1304] [ndcg@20: 0.2162]
