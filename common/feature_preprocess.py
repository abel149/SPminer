import os
import pickle
import random
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, PPI, QM9
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
import queue
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph
#import orca
from torch_scatter import scatter_add

from common import utils

AUGMENT_METHOD = "concat"
FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = [], []
#FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS = ["identity"], [4]
#FEATURE_AUGMENT = ["motif_counts"]
#FEATURE_AUGMENT_DIMS = [73]
#FEATURE_AUGMENT_DIMS = [15]

def norm(edge_index, num_nodes, edge_weight=None, improved=False,
         dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def compute_identity(edge_index, n, k):
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float,
                             device=edge_index.device)
    edge_index, edge_weight = pyg_utils.add_remaining_self_loops(
        edge_index, edge_weight, 1, n)
    adj_sparse = torch.sparse.FloatTensor(edge_index, edge_weight,
        torch.Size([n, n]))
    adj = adj_sparse.to_dense()

    deg = torch.diag(torch.sum(adj, -1))
    deg_inv_sqrt = deg.pow(-0.5)
    adj = deg_inv_sqrt @ adj @ deg_inv_sqrt 

    diag_all = [torch.diag(adj)]
    adj_power = adj
    for i in range(1, k):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    diag_all = torch.stack(diag_all, dim=1)
    return diag_all
FEATURE_AUGMENT = ["node_degree", "pagerank"]
FEATURE_AUGMENT_DIMS = [8, 1]  # Example dims for enabled features

class FeatureAugment(nn.Module):
    def __init__(self):
        super(FeatureAugment, self).__init__()

        def degree_fun(graph, feature_dim):
            graph.node_degree = self._one_hot_tensor(
                [d for _, d in graph.G.degree()],
                one_hot_dim=feature_dim)
            return graph
        


        def pagerank_fun(graph, feature_dim):
            nodes = list(graph.G.nodes)
            pagerank = nx.pagerank(graph.G)
            graph.pagerank = torch.tensor([pagerank[x] for x in nodes]).unsqueeze(1)
            return graph

        def node_features_base_fun(graph, feature_dim):
            for v in graph.G.nodes:
                if "node_feature" not in graph.G.nodes[v]:
                    graph.G.nodes[v]["node_feature"] = torch.ones(feature_dim)
            return graph

        self.node_features_base_fun = node_features_base_fun
        self.node_feature_funs = {
            "node_degree": degree_fun,
            "pagerank": pagerank_fun
        }

    def register_feature_fun(self, name, feature_fun):
        self.node_feature_funs[name] = feature_fun
        


    @staticmethod
    def _one_hot_tensor(list_scalars, one_hot_dim=1):
        if not isinstance(list_scalars, list) and not list_scalars.ndim == 1:
            raise ValueError("input to _one_hot_tensor must be 1-D list")
        vals = torch.LongTensor(list_scalars).view(-1,1)
        vals = vals - torch.min(vals)
        vals = torch.min(vals, torch.tensor(one_hot_dim - 1))
        vals = torch.max(vals, torch.tensor(0))
        one_hot = torch.zeros(len(list_scalars), one_hot_dim)
        one_hot.scatter_(1, vals, 1.0)
        return one_hot

    def augment(self, dataset):
        dataset = dataset.apply_transform(self.node_features_base_fun, feature_dim=1)
        for key, dim in zip(FEATURE_AUGMENT, FEATURE_AUGMENT_DIMS):
            dataset = dataset.apply_transform(self.node_feature_funs[key], feature_dim=dim)
        return dataset

class Preprocess(nn.Module):
    def __init__(self, dim_in):
        super(Preprocess, self).__init__()
        self.dim_in = dim_in
        if AUGMENT_METHOD == 'add':
            self.module_dict = {
                    key: nn.Linear(aug_dim, dim_in)
                    for key, aug_dim in zip(FEATURE_AUGMENT, 
                                            FEATURE_AUGMENT_DIMS)
                    }

    @property
    def dim_out(self):
        if AUGMENT_METHOD == 'concat':
            return self.dim_in + sum(
                    [aug_dim for aug_dim in FEATURE_AUGMENT_DIMS])
        elif AUGMENT_METHOD == 'add':
            return dim_in
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                    AUGMENT_METHOD))

    def forward(self, batch):
        if AUGMENT_METHOD == 'concat':
            feature_list = [batch.node_feature]
            for key in FEATURE_AUGMENT:
                feature_list.append(batch[key])
            batch.node_feature = torch.cat(feature_list, dim=-1)
        elif AUGMENT_METHOD == 'add':
            for key in FEATURE_AUGMENT:
                batch.node_feature = batch.node_feature + self.module_dict[key](
                        batch[key])
        else:
            raise ValueError('Unknown feature augmentation method {}.'.format(
                    AUGMENT_METHOD))
        return batch
