import random
from dgl.subgraph import node_subgraph
import pandas as pd
import numpy as np
import torch


class DataLoader:
    def __init__(self, data, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True):
        self.data = data
        self.n_nodes = n_nodes
        self.neg = neg
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.shuffle = shuffle
        self.params = {'batch_size': batch_size, 'neg_ratio': neg_ratio}

    def __len__(self):
        if self.neg is not None:
            return (len(self.data) + len(self.neg)) // self.batch_size
        return len(self.data) * (1 + self.neg_ratio) // self.batch_size

    def __iter__(self):
        epoch_data = self._prepare_data()
        bs = self.batch_size
        for i in range(len(epoch_data) // self.batch_size):
            batch = epoch_data[i*bs:(i+1)*bs].reset_index(drop=True)
            yield self._generate_batch(batch)

    def _prepare_data(self):
        # negative sampling
        if self.neg is None:
            epoch_data = pd.concat([self.data, self._neg_sample()], ignore_index=True)
        else:
            epoch_data = pd.concat([self.data, self.neg], ignore_index=True)
        
        # shuffle data
        if self.shuffle:
            epoch_data = epoch_data.sample(frac=1).reset_index(drop=True)
        
        return epoch_data

    def _neg_sample(self):
        neg_src = np.repeat(self.data['src_ind'], self.neg_ratio)

        dst = list(set(self.data['dst_ind']))

        # interacted = self.data.groupby(self.data['src_ind']).apply(lambda x: set(x['dst_ind']))
        # neg_dst = [random.choice(list(dst - interacted[src])) for src in neg_src]
        neg_dst = [random.choice(dst) for _ in range(len(neg_src))]
        # neg_dst = np.random.randint(0, self.n_nodes, len(neg_src))
        neg_df = pd.DataFrame({'src_ind': neg_src, 'dst_ind': neg_dst, 'label': 0, 'date': np.nan})
        return neg_df

    def _generate_batch(self, batch):
        return batch


class BlockSamplingDataLoader(DataLoader):
    def __init__(self, data, graph, block_sampler, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True):
        super().__init__(data, n_nodes=n_nodes, neg=neg, batch_size=batch_size, neg_ratio=neg_ratio, shuffle=shuffle)

        self.graph = graph
        self.block_sampler = block_sampler

    def _generate_batch(self, batch):
        all_nodes = list(batch['src_ind']) + list(batch['dst_ind'])
        seed_nodes, out_ind, node_ind = [], [], {}
        for k in all_nodes:
            if k not in node_ind:
                node_ind[k] = len(seed_nodes)
                seed_nodes.append(k)

            out_ind.append(node_ind[k])

        seed_nodes = torch.tensor(seed_nodes)
        mfg = self.block_sampler.sample_blocks(self.graph, seed_nodes)
        return batch, (mfg, out_ind)


class BlockSeqSamplingDataLoader(DataLoader):
    def __init__(self, data, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True):
        super().__init__(data, n_nodes=n_nodes, neg=neg, batch_size=batch_size, neg_ratio=neg_ratio, shuffle=shuffle)
