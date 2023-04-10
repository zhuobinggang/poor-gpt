import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
from dgl.nn import GraphConv
import numpy as np
import scipy.sparse as sp

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
# Split edge set for training and testing
u, v = g.edges()

eids = np.arange(g.number_of_edges()) # (10556)
eids = np.random.permutation(eids) # (10556)
test_size = int(len(eids) * 0.1) # (1055)
train_size = g.number_of_edges() - test_size # (9501)
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]] # ()
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
# u最大到2706，v也同样。代表了node的下标
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))) # (2708, 2708)
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes()) # 
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = (
    neg_u[neg_eids[:test_size]],
    neg_v[neg_eids[:test_size]],
)
train_neg_u, train_neg_v = (
    neg_u[neg_eids[test_size:]],
    neg_v[neg_eids[test_size:]],
)
