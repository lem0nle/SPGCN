# RGCN with heterogeneous graph feature for recommendation

import pandas as pd
from datetime import datetime
from loguru import logger

from invest.utils import DataLoader, build_hetero_graph, load_data, dump_result, evaluate, make_path, print_metrics
from invest.model.HGCN import RGCN

path = 'data/tyc/'
save_path = f'ws/RGCN/{datetime.now().strftime("%Y%m%d%H%M%S")}/'
make_path(save_path)

logger.add(save_path + 'train.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
logger.info('******* RGCN *******')

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
model = RGCN(graph=g, in_feats=64, hid_feats=64, out_feats=64, n_nodes=n_nodes, rel_names=edge_dfs.keys())
logger.info(model.params)

# build data loader
train_loader = DataLoader(train, batch_size=10000)
logger.info(train_loader.params)

# train and save model
logger.info('training...')
params = {'epoch': 1, 'lr': 5e-3}
model.fit(train_loader, test, test_neg, epoch=params['epoch'], lr=params['lr'])
model.save(save_path + f'model_{params["epoch"]}_{params["lr"]}.snapshot')

logger.info('training finished')

# predict
# model.load(save_path + f"model_{params['epoch']}_{params['lr']}.snapshot")
logger.info('predicting...')
pred = model.predict(test)
pred_neg = model.predict(test_neg)
pred = pd.concat([pred, pred_neg], ignore_index=True)
pred = pred.sample(frac=1).reset_index(drop=True)
dump_result(pred, save_path + f'result_{params["epoch"]}_{params["lr"]}.csv')

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