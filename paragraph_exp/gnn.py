import torch
from torch import nn
from torch import optim
from torch_geometric.data import Data
from data_loader import loader
from itertools import chain
import fasttext
from fugashi import Tagger
from common import combine_ss, ModelWrapper, train_save_eval_plot
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GCN
from torch_geometric.utils import add_self_loops, degree

def edge_create(length):
    edge_index_raw = []
    for i in range(0, length - 1):
        edge_index_raw.append([i, i+1])
        edge_index_raw.append([i+1, i])
        if i+2 < length:
            edge_index_raw.append([i, i+2])
            edge_index_raw.append([i+2, i])
    return torch.tensor(edge_index_raw, dtype=torch.long)

def create_data(x, global_node = False):
    edge_index = edge_create(len(x))
    if global_node:
        LENGTH = x.size(0)
        zeros = torch.zeros(1, 300)
        x = torch.cat((zeros, x)) # (? + 1 , 300)
        assert x.size(0) == LENGTH + 1
        raw = []
        for i in range(LENGTH):
            raw.append([i, LENGTH])
            raw.append([LENGTH, i])
        raw = torch.tensor(raw, dtype=torch.long)
        assert raw.size(0) == LENGTH * 2
        assert edge_index.size(1) == 2
        edge_index = torch.cat((edge_index, raw))
    data = Data(x=x, edge_index=edge_index.t())
    data.validate(raise_on_error=True)
    return data



############## GCN ################

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_() # NOTE: 将b设定为zero是常态吗？
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # NOTE: 自循环。。不太了解
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Compute normalization.
        row, col = edge_index 
        deg = degree(col, x.size(0), dtype=x.dtype) # NOTE: degree是什么意思？根据进来的边的数量进行norm么
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        # Step 6: Apply a final bias vector.
        out += self.bias
        return out
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    tagger = Tagger('-Owakati')
    ld_train = loader.news.train()
    ss, ls = ld_train[0]
    left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(combine_ss(ss[:2]))]) # (?, 300)
    right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(combine_ss(ss[2:]))]) # (?, 300)
    # Left graph
    left_data = create_data(left_vecs)
    left_data.validate(raise_on_error=True)
    # Right graph
    right_data = create_data(right_vecs)
    right_data.validate(raise_on_error=True)
def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    tagger = Tagger('-Owakati')
    ld_train = loader.news.train()
    ss, ls = ld_train[0]
    left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(combine_ss(ss[:2]))]) # (?, 300)
    right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(combine_ss(ss[2:]))]) # (?, 300)
    # Left graph
    left_data = create_data(left_vecs)
    # Right graph
    right_data = create_data(right_vecs)
    # Train
    conv = GCNConv(300, 300)


# 双方向 & Attention & GCN
class Sector_GCN(nn.Module):
    def __init__(self, learning_rate = 1e-3):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.gcn1 = GCN(300, 100, 2, 100).cuda()
        self.rnn = nn.LSTM(100, 100, 1, batch_first = True, bidirectional = True).cuda()
        self.attention = nn.Sequential(
            nn.Linear(200, 1),
            nn.Softmax(dim = 0),
        ).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(400, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.gcn1.parameters(), self.rnn.parameters(), self.mlp.parameters(), self.attention.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        mlp = model.mlp
        vec_cat = model.get_pooled_output(ss, ls) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def get_pooled_output(model, ss, ls):
        rnn = model.rnn
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        # GCN
        left_g = create_data(left_vecs)
        right_g = create_data(right_vecs)
        left_vecs = model.gcn1(left_g.x.cuda(), left_g.edge_index.cuda()) # (? 100)
        right_vecs = model.gcn1(right_g.x.cuda(), right_g.edge_index.cuda()) # (?, 100)
        assert len(left_vecs.shape) == 2 
        assert left_vecs.shape[1] == 100
        # RNN
        left_vecs, (_, _) = rnn(left_vecs) # (?, 200)
        right_vecs, (_, _) = rnn(right_vecs) # (?, 200)
        assert len(left_vecs.shape) == 2 
        assert left_vecs.shape[1] == 200
        # Attention
        left_vecs = model.attention(left_vecs) * left_vecs # (?, 200)
        right_vecs = model.attention(right_vecs) * right_vecs # (?, 200)
        left_vec = left_vecs.sum(dim = 0) # (600)
        right_vec = right_vecs.sum(dim = 0) # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        return vec_cat

def script():
    model = ModelWrapper(Sector_GCN())
    train_save_eval_plot(model, 'GCN', batch_size = 32, check_step = 1000, total_step = 10000)


# 双方向 & Attention & GCN
class GCN_Only(nn.Module):
    def __init__(self, in_channel, out_channel, layer, learning_rate = 1e-3):
        super().__init__()
        self.out_channel = out_channel
        self.ft = fasttext.load_model('cc.ja.300.bin')
        # self.gcn = GCN(300, 50, 3, 50).cuda() # 20352 paras
        self.gcn = GCN(in_channel, out_channel, layer, out_channel).cuda()
        self.mlp = nn.Linear(out_channel * 2, 2).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.gcn.parameters(), self.mlp.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        mlp = model.mlp
        vec_cat = model.get_pooled_output(ss, ls) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def get_pooled_output(model, ss, ls):
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        # GCN
        left_g = create_data(left_vecs, global_node = True)
        right_g = create_data(right_vecs, global_node = True)
        left_vecs = model.gcn(left_g.x.cuda(), left_g.edge_index.cuda()) # (? 100)
        right_vecs = model.gcn(right_g.x.cuda(), right_g.edge_index.cuda()) # (?, 100)
        assert len(left_vecs.shape) == 2 
        assert left_vecs.shape[1] == model.out_channel
        # Attention
        left_vec = left_vecs[-1] # (50)
        right_vec = right_vecs[-1] # (50)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (100)
        return vec_cat

def script():
    # model = ModelWrapper(GCN_Only(300, 50, 3))
    model = ModelWrapper(GCN_Only(300, 100, 2))
    train_save_eval_plot(model, 'GCNonly300X100X2', batch_size = 32, check_step = 1000, total_step = 10000)
