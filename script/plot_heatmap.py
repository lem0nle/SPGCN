from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger
import dgl
from invest.utils import build_hetero_graph, load_data
from invest.dataloader import CSNetSamplingDataLoader
from invest.model.CSNet import CSNet
import torch
import matplotlib
import matplotlib.pyplot as plt
import random
import yaml
from yaml import Loader

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

random.seed(523)
torch.random.manual_seed(123)

path = 'data/tyc/'

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
model = CSNet(graph=g, in_feats=16, hid_feats=8, out_feats=8, n_nodes=n_nodes, rel_names=g.etypes)
logger.info(model.params)

block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_loader = CSNetSamplingDataLoader(train, g, big, block_sampler, neg_ratio=50, batch_size=50000)
test_loader = CSNetSamplingDataLoader(test, g, big, block_sampler, neg=test_neg, batch_size=100000)

# train model
model.fit(train_loader, test_loader, epoch=1, weight_decay=1e-2, lr=1e-3, print_every=10)
model.save('model/heatmap-2.snapshot')

# or: load model
# model.load('model/heatmap-2.snapshot')

# get comp info
info = pd.read_csv('data/tyc/comp_info.csv')
pos = pd.read_csv('data/tyc/comp_pos.csv')['pos']

# select ids
# comps = pd.read_csv(path + 'comp_info.csv')
# n_users = test['src_ind'].max()
# with_ind = comps.dropna().index
# users = with_ind[with_ind < n_users]
# items = with_ind[with_ind >= n_users]
users = train['src_ind'][2:3]
items = train.groupby('src_ind')['dst_ind'].apply(list)[users[2]]
# items = train['dst_ind'][:10]

trans = yaml.load(open('data/trans.yml'), Loader=Loader)

# get embeddings
torch.set_grad_enabled(False)
_, mfg, bimfg, lmfg, out_ind = train_loader._generate_batch({'src_ind': users, 'dst_ind': items})
hcf, hc, hs = model._pred(mfg, bimfg, lmfg, out_ind)
eu = hcf[:len(users)]
ei = hcf[len(users):]
cf = hc[len(users):]
sf = hs[len(users):]

cf_by_industry = defaultdict(list)
sf_by_industry = defaultdict(list)

for i in range(len(ei)):
    if isinstance(info['industry'][items[i]], float):
        continue
    bc = np.dot(eu[0], cf[i])
    bs = np.dot(eu[0], sf[i])
    alphac = np.exp(bc) / (np.exp(bc) + np.exp(bs))
    alphas = np.exp(bs) / (np.exp(bc) + np.exp(bs))
    print(','.join(trans[x] for x in info['industry'][items[i]].split(':')), alphac, alphas)
    for x in info['industry'][items[i]].split(':'):
        cf_by_industry[trans[x]].append(cf[i])
        sf_by_industry[trans[x]].append(sf[i])

N = 24
skip = 1

print(cf_by_industry)
keys = list(sorted(cf_by_industry.keys()))
vegetables = ['Housing', 'IoT', 'E-Commerce', 'Enterprise-Service', 'Advanced-Manufacturing', 'Environment', 'Construction', 'Traveling', 'Vehicle', 'Social', 'Software', 'Hardware', 'Gaming']
farmers = vegetables
harvest = np.zeros((len(vegetables), len(farmers)))

# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         s = 0
#         for k in cf_by_industry[vegetables[i]]:
#             for l in cf_by_industry[farmers[j]]:
#                 s += torch.cosine_similarity(k, l, dim=0)
#         harvest[i, j] = s / len(cf_by_industry[vegetables[i]]) / len(cf_by_industry[farmers[j]])

for i in range(len(vegetables)):
    for j in range(len(farmers)):
        s = 0
        for k in sf_by_industry[vegetables[i]]:
            for l in sf_by_industry[farmers[j]]:
                s += torch.cosine_similarity(k, l, dim=0)
        harvest[i, j] = s / len(sf_by_industry[vegetables[i]]) / len(sf_by_industry[farmers[j]])



def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin=0, vmax=1, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


fig, ax = plt.subplots()

im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="afmhot", cbarlabel="similarity")

fig.tight_layout()
# plt.show()
plt.savefig('heatmap-sub.pdf')


# im = ax.imshow(harvest)

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(farmers)), labels=farmers)
# ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# # for i in range(len(vegetables)):
# #     for j in range(len(farmers)):
# #         text = ax.text(j, i, harvest[i, j],
# #                        ha="center", va="center", color="w")

# # ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()

'''
# plot
# plt.scatter(eu[:,0], eu[:,1])
c = -np.log(pos[items] + 0.1)
plt.xlim(1.2, 2.4)
plt.ylim(1.2, 2.2)
# plt.xlim(1.984, 2.01)
# plt.ylim(1.95, 2.01)

plt.scatter(cf[:,0], cf[:,1], c=c, s=64, cmap='GnBu')
plt.scatter(sf[:,0], sf[:,1], marker='^', c=c, s=64, cmap='copper')

for i, ind in enumerate(items):
    # i = i + 100
    # if random.random() > 0.3: continue
    label = info['industry'][ind]
    if isinstance(label, str):
        labels = list(set(label.split(':')))
        label = trans[labels[0]]
        if label == 'Enterprise-Service': continue
        if label == 'Producing': continue
        if label == 'Finance': continue
        plt.annotate(label, (cf[i, 0], cf[i, 1] + (len(label) - 7) * 0.001), alpha=0.8, usetex=True)
        plt.annotate(label, (sf[i, 0], sf[i, 1]), alpha=0.8, usetex=True)

# plt.show()
plt.savefig('sub-cluster-2.pdf')


# plt.xlim(1.984, 2.01)
# plt.ylim(1.95, 2.01)

# plt.scatter(cf[:,0], cf[:,1], c=c, s=64, cmap='GnBu')
# plt.scatter(sf[:,0], sf[:,1], marker='^', c=c, s=64, cmap='copper')

# trans = yaml.load(open('data/trans.yml'), Loader=Loader)

# for i, ind in enumerate(items[:200]):
#     # i = i + 100
#     label = info['industry'][ind]
#     if isinstance(label, str):
#         labels = list(set(label.split(':')))
#         label = trans[labels[0]]
#         if label == 'Enterprise-Service': continue
#         if label == 'Producing': continue
#         if label == 'Finance': continue
#         plt.annotate(label, (cf[i, 0], cf[i, 1] + (len(label) - 7) * 0.001))
#         plt.annotate(label, (sf[i, 0], sf[i, 1]))

# # plt.show()
# plt.savefig('com-cluster-2.pdf')
'''
