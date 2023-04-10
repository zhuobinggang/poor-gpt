import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
# Create the model with given dimensions
model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0
    features = g.ndata["feat"] # (2708, 1433)
    labels = g.ndata["label"] # (2708)
    train_mask = g.ndata["train_mask"] # (2708) mask是用来区分train, val和test的
    val_mask = g.ndata["val_mask"] # 同上
    test_mask = g.ndata["test_mask"] # 同上
    for e in range(100):
        # Forward
        logits = model(g, features) # (2708, 7)
        # Compute prediction
        pred = logits.argmax(1) # (2708)
        # Compute loss
        train_logits = logits[train_mask] # 挑选出train中有的节点 -> (140, 7)
        train_labels = labels[train_mask] # (140)
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(train_logits, train_labels) 
        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )



