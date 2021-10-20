from collections import deque
import random
import dgl
from dgl.sampling.neighbor import sample_neighbors
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
            l = len(self.data) + len(self.neg)
        else:
            l = len(self.data) * (1 + self.neg_ratio)
        return (l + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        epoch_data = self._prepare_data()
        bs = self.batch_size
        for i in range((len(epoch_data) + self.batch_size - 1) // self.batch_size):
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
    def __init__(self, data, graph, block_sampler, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True, separate_label_graph=False):
        super().__init__(data, n_nodes=n_nodes, neg=neg, batch_size=batch_size, neg_ratio=neg_ratio, shuffle=shuffle)

        # self.graph = graph
        self.graph = graph
        self.separate_label_graph = separate_label_graph
        self.label_graph = graph.edge_type_subgraph(['label']) if separate_label_graph else None
        self.block_sampler = block_sampler

    def _generate_batch(self, batch):
        all_nodes = list(batch['src_ind']) + list(batch['dst_ind'])
        mfg, out_ind = self._generate_mfg(self.graph, all_nodes)
        if self.separate_label_graph:
            lmfg, lout_ind = self._generate_mfg(self.label_graph, all_nodes)
        return (batch, (mfg, out_ind), (lmfg, lout_ind)) if self.separate_label_graph else (batch, (mfg, out_ind))

    def _generate_mfg(self, graph, all_nodes):
        seed_nodes, out_ind, node_ind = [], [], {}
        for k in all_nodes:
            if k not in node_ind:
                node_ind[k] = len(seed_nodes)
                seed_nodes.append(k)

            out_ind.append(node_ind[k])

        seed_nodes = torch.tensor(seed_nodes)
        mfg = self.block_sampler.sample_blocks(graph, seed_nodes)
        return mfg, out_ind


class BlockSeqSamplingDataLoader(BlockSamplingDataLoader):
    def __init__(self, data, graph, block_sampler, seq_sampler, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True, separate_label_graph=False):
        super().__init__(data, graph, block_sampler, n_nodes=n_nodes, neg=neg, batch_size=batch_size, neg_ratio=neg_ratio, shuffle=shuffle, separate_label_graph=separate_label_graph)
        self.seq_sampler = seq_sampler

    def _generate_batch(self, batch):
        all_nodes = list(batch['src_ind']) + list(batch['dst_ind'])

        mfg, out_ind = self._generate_mfg(self.graph, all_nodes)
        seqs = self._generate_seq(all_nodes)

        if self.separate_label_graph:
            lmfg, lout_ind = self._generate_mfg(self.label_graph, all_nodes)

        return (batch, (mfg, out_ind), (lmfg, lout_ind), seqs) if self.separate_label_graph else (batch, (mfg, out_ind), seqs)
    
    def _generate_seq(self, all_nodes):
        return self.seq_sampler.sample_seqs(self.graph, all_nodes)


class RandomWalkSampler:
    def __init__(self, seq_len):
        self.seq_len = seq_len
    
    def sample_seqs(self, g, nodes):
        seq = deque([nodes])
        # forward
        for _ in range((self.seq_len - 1) // 2):
            neighbors = []
            sg = sample_neighbors(g, nodes, 1, edge_dir='out')
            src, dst = list(zip(*[sg.edges(etype=etype) for etype in sg.etypes]))
            src = torch.cat(src).tolist()
            dst = torch.cat(dst).tolist()
            dst2src = {d: s for s, d in zip(src, dst)}
            neighbors = [dst2src.get(n, n) for n in nodes]
            seq.append(neighbors)
            nodes = neighbors
        nodes = seq[0]
        # backward
        for _ in range((self.seq_len - 1) // 2):
            neighbors = []
            sg = sample_neighbors(g, nodes, 1, edge_dir='in')
            src, dst = list(zip(*[sg.edges(etype=etype) for etype in sg.etypes]))
            src = torch.cat(src).tolist()
            dst = torch.cat(dst).tolist()
            dst2src = {d: s for s, d in zip(src, dst)}
            neighbors = [dst2src.get(n, n) for n in nodes]
            seq.appendleft(neighbors)
            nodes = neighbors
        return torch.tensor(seq)


class CSNetSamplingDataLoader(BlockSamplingDataLoader):
    def __init__(self, data, graph, bigraph, block_sampler, n_nodes=None, neg=None, batch_size=32, neg_ratio=4, shuffle=True):
        super().__init__(data, graph, block_sampler, n_nodes=n_nodes, neg=neg, batch_size=batch_size, neg_ratio=neg_ratio, shuffle=shuffle, separate_label_graph=True)
        self.bigraph = dgl.to_homogeneous(bigraph)
        self.label_graph = dgl.add_self_loop(dgl.to_homogeneous(self.label_graph, store_type=False))

    def _generate_batch(self, batch):
        all_nodes = list(batch['src_ind']) + list(batch['dst_ind'])
        mfg, out_ind = self._generate_mfg(self.graph, all_nodes)
        bimfg, _ = self._generate_mfg(self.bigraph, all_nodes)
        lmfg, _ = self._generate_mfg(self.label_graph, all_nodes)
        return batch, mfg, bimfg, lmfg, out_ind
