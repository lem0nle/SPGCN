# GCN for recommendation

import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from invest.utils import build_graph, format_metrics, load_data, dump_result, evaluate, make_path
from invest.dataloader import DataLoader
from invest.model.GCN import GCN


path = 'data/tyc/'
save_path = f'ws/GCN/{datetime.now().strftime("%Y%m%d%H%M%S")}/'
make_path(save_path)

logger.add(save_path + 'train.log', format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
logger.info('******* GCN *******')

# load raw data
logger.info('loading data')
train = load_data(path + 'train.csv')
valid = load_data(path + 'valid.csv')
train = pd.concat([train, valid], ignore_index=True)

test = load_data(path + 'test.csv')
test_neg = load_data(path + 'test_neg.csv')
n_nodes = np.concatenate([train['src_ind'], train['dst_ind'], test['src_ind'], test['dst_ind']]).max() + 1

# build graph
g = build_graph(train, n_nodes)
# g = build_graph(pd.concat([train, test], ignore_index=True), n_nodes)
logger.info(g)

# build model
logger.info('building model')
model = GCN(graph=g, in_feats=64, out_feats=64, n_nodes=n_nodes)
logger.info(model.params)

# build data loader
train_loader = DataLoader(train, n_nodes, batch_size=80000, neg_ratio=50)
# test_loader = DataLoader(test, n_nodes, batch_size=256, neg=test_neg, shuffle=False)
logger.info(train_loader.params)

# train and save model
logger.info('training...')
params = {'epoch': 30, 'lr': 1e-2}
model.fit(train_loader, test, test_neg, epoch=params['epoch'], lr=params['lr'], print_every=1)
model.save(save_path + f'model_{params["epoch"]}_{params["lr"]}.snapshot')

logger.info('training finished')

# predict
model.load(save_path + f'model_{params["epoch"]}_{params["lr"]}.snapshot')
logger.info('predicting...')
pred = model.predict(test)
pred_neg = model.predict(test_neg)
pred = pd.concat([pred, pred_neg], ignore_index=True)
pred = pred.sample(frac=1).reset_index(drop=True)
dump_result(pred, save_path + f'result_{params["epoch"]}_{params["lr"]}.csv')

# evaluate
metrics = evaluate(test, pred, top_k=[5, 10, 20])
logger.info(format_metrics(metrics))

# best:
# {'precision@5': 0.011764705882352938, 'recall@5': 0.03613709020605105, 'ndcg@5': 0.025209521323411817, 'map@5': 0.017221131468785882}
# {'precision@10': 0.012790697674418601, 'recall@10': 0.07629542419686673, 'ndcg@10': 0.03909899744081252, 'map@10': 0.022542016817142775}
# {'precision@20': 0.013337893296853625, 'recall@20': 0.15492285064355768, 'ndcg@20': 0.061127759186525445, 'map@20': 0.02811449181881521}

# {'precision@5': 0.09166666666666666, 'recall@5': 0.28872697434858274, 'ndcg@5': 0.1996352716785345, 'map@5': 0.14606501878085382}
# {'precision@10': 0.07275641025641028, 'recall@10': 0.40168653644477575, 'ndcg@10': 0.24312913053544, 'map@10': 0.16549381866383794}
# {'precision@20': 0.05423076923076923, 'recall@20': 0.519906906756495, 'ndcg@20': 0.28187889864635224, 'map@20': 0.17835464278454222}

# [mae: 0.3539] [rmse: 0.3912] [auc: 0.6452]
# [precision@5: 0.0626] [recall@5: 0.2043] [map@5: 0.0999] [ndcg@5: 0.1381]
# [precision@10: 0.0556] [recall@10: 0.3146] [map@10: 0.1174] [ndcg@10: 0.1797]
# [precision@20: 0.0407] [recall@20: 0.4022] [map@20: 0.1263] [ndcg@20: 0.2082]
