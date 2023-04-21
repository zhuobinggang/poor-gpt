import torch
from torch import nn
from torch import optim
from torch_geometric.data import Data
from data_loader import loader
from itertools import chain
import fasttext
from fugashi import Tagger
from common import combine_ss, ModelWrapper, train_save_eval_plot, parameters
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GCN
from torch_geometric.utils import add_self_loops, degree
import math

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
        zeros = torch.zeros(1, 300).cuda()
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
    def create_graph_by_text(self, text):
        tagger = self.tagger
        ft = self.ft
        vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(text)]) # (?, 300)
        g = create_data(vecs, global_node = True)
        return g
    def get_pooled_output(model, ss, ls):
        left_g = model.create_graph_by_text(combine_ss(ss[:2]))
        right_g = model.create_graph_by_text(combine_ss(ss[2:]))
        # GCN
        left_vecs = model.gcn(left_g.x.cuda(), left_g.edge_index.cuda()) # (? 100)
        right_vecs = model.gcn(right_g.x.cuda(), right_g.edge_index.cuda()) # (?, 100)
        assert len(left_vecs.shape) == 2 
        assert left_vecs.size(1) == model.out_channel
        # Attention
        left_vec = left_vecs[-1] # (50)
        right_vec = right_vecs[-1] # (50)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (100)
        return vec_cat

def script():
    # model = ModelWrapper(GCN_Only(300, 50, 3))
    model = ModelWrapper(GCN_Only(300, 100, 2))
    train_save_eval_plot(model, 'GCNonly300X100X2', batch_size = 32, check_step = 1000, total_step = 10000)


###### 试试单纯注意力机制 #######

# 双方向 & Attention & GCN
class Att_Only(nn.Module):
    def __init__(self,learning_rate = 1e-3, hidden_channel = 1800):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.att1 = nn.Sequential(
            nn.Linear(300, 1),
            nn.Softmax(dim = 0),
        ).cuda()
        self.att2 = nn.Sequential(
            nn.Linear(300, 1),
            nn.Softmax(dim = 0),
        ).cuda()
        self.att3 = nn.Sequential(
            nn.Linear(300, 1),
            nn.Softmax(dim = 0),
        ).cuda()
        self.mlp1 = nn.Sequential(
            nn.Linear(300, 50),
            nn.GELU(),
            nn.Linear(50, 300),
        ).cuda()
        self.hidden_channel = hidden_channel
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channel, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.att1.parameters(), self.att2.parameters(), self.att3.parameters(), self.mlp1.parameters(), self.mlp2.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
        # postion
        self.position_encoding_init()
    def position_encoding_init(self):
        max_len = 500
        d_model = 300
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda() # (500, 1 , 300)
    def forward(model, ss, ls):
        mlp = model.mlp2
        vec_cat = model.get_pooled_output(ss, ls) # (1800)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def attention_my_self(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]).cuda() # (?, 300)
        nonlinear = self.mlp1(vecs) #NOTE: 事实证明这里的非线性层非常重要，不知道为什么……
        att1 = (self.att1(vecs) * nonlinear).mean(dim = 0) # (300)
        att2 = (self.att2(vecs) * nonlinear).mean(dim = 0)
        att3 = (self.att3(vecs) * nonlinear).mean(dim = 0)
        out = torch.cat((att1, att2, att3))
        assert out.size(0) == 900
        return out
    def get_pooled_output(model, ss, ls):
        left = model.attention_my_self(combine_ss(ss[:2]))  # (900)
        right = model.attention_my_self(combine_ss(ss[2:]))  # (900)
        # Attention
        vec_cat = torch.cat((left, right)) # (1800)
        assert len(vec_cat.shape) == 1
        assert vec_cat.size(0) == model.hidden_channel
        return vec_cat

class Att_Pos(Att_Only):
    def attention_my_self(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]).cuda() # (?, 300)
        # POS Encoding
        pe = self.pe[:vecs.size(0), 0, :] # (?, 300)
        vecs = vecs + pe
        nonlinear = self.mlp1(vecs) #NOTE: 事实证明这里的非线性层非常重要，不知道为什么……
        # nonlinear = self.mlp1(vecs) # (?, 300)
        att1 = (self.att1(vecs) * nonlinear).mean(dim = 0) # (300)
        att2 = (self.att2(vecs) * nonlinear).mean(dim = 0)
        att3 = (self.att3(vecs) * nonlinear).mean(dim = 0)
        out = torch.cat((att1, att2, att3))
        assert out.size(0) == 900
        return out

class Att_Pos_Mean(Att_Pos):
    def __init__(self,learning_rate = 1e-3):
        super().__init__(hidden_channel = 600)
    def attention_my_self(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]).cuda() # (?, 300)
        # POS Encoding
        pe = self.pe[:vecs.size(0), 0, :] # (?, 300)
        vecs = vecs + pe
        nonlinear = self.mlp1(vecs) #NOTE: 事实证明这里的非线性层非常重要，不知道为什么……
        # nonlinear = self.mlp1(vecs) # (?, 300)
        att1 = (self.att1(vecs) * nonlinear).mean(dim = 0) # (300)
        att2 = (self.att2(vecs) * nonlinear).mean(dim = 0)
        att3 = (self.att3(vecs) * nonlinear).mean(dim = 0)
        out = torch.stack((att1, att2, att3)).mean(0)
        assert out.size(0) == 300
        return out

def script():
    model = ModelWrapper(Att_Pos_Mean())
    train_save_eval_plot(model, 'ATT_POS_mean', batch_size = 32, check_step = 1000, total_step = 10000)


########## GCN with pos #############

class GCN_POS(GCN_Only):
    def __init__(self, in_channel, out_channel, layer, learning_rate = 1e-3):
        super().__init__(in_channel, out_channel, layer, learning_rate = 1e-3)
        self.position_encoding_init()
    def position_encoding_init(self):
        max_len = 500
        d_model = 300
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda() # (500, 1 , 300)
    def create_graph_by_text(self, text):
        tagger = self.tagger
        ft = self.ft
        vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(text)]) # (?, 300)
        # NOTE: ADD POS INFO
        vecs = vecs.cuda() + self.pe[:vecs.size(0), 0, :] # (?, 300)
        g = create_data(vecs, global_node = True)
        return g

def script():
    model = ModelWrapper(GCN_POS(300, 50, 3))
    train_save_eval_plot(model, 'GCN_POS', batch_size = 32, check_step = 1000, total_step = 10000)


#################### GCN Unique Word ##################


def create_if_not_exist(dic, word, vec):
    if word not in dic:
        dic[word] = {'id': len(dic.keys()), 'vec': vec, 'adj': set()}

def connect(dic, w1, w2):
    dic[w1]['adj'].add(dic[w2]['id'])
    dic[w2]['adj'].add(dic[w1]['id'])

# NOTE: 未完成
class GCN_TextING(GCN_Only):
    def create_graph_by_text(self, text):
        tagger = self.tagger
        ft = self.ft
        words = [str(word) for word in tagger(text)]
        # Slide window and create graph
        dic = {}
        for w1, w2, w3 in zip(words[0::3], words[1::3], words[2::3]):
            _ = [create_if_not_exist(dic, w, torch.from_numpy(ft.get_word_vector(w))) for w in [w1, w2, w3]]
            connect(dic, w1, w2)
            connect(dic, w2, w3)
            connect(dic, w1, w3)
        # TODO
        vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(text)]) # (?, 300)
        g = create_data(vecs, global_node = True)
        return g
    def get_pooled_output(model, ss, ls):
        left_g = model.create_graph_by_text(combine_ss(ss[:2]))
        right_g = model.create_graph_by_text(combine_ss(ss[2:]))
        # GCN
        left_vecs = model.gcn(left_g.x.cuda(), left_g.edge_index.cuda()) # (? 100)
        right_vecs = model.gcn(right_g.x.cuda(), right_g.edge_index.cuda()) # (?, 100)
        assert len(left_vecs.shape) == 2 
        assert left_vecs.size(1) == model.out_channel
        # Attention
        left_vec = left_vecs[-1] # (50)
        right_vec = right_vecs[-1] # (50)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (100)
        return vec_cat


############# Adapter #############

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(300, 10)
        self.gelu = nn.GELU()
        self.up = nn.Linear(10, 300)
    def forward(self, x):
        h = self.down(x) 
        h = self.gelu(h)
        h = self.up(h)
        return x + h

# NOTE: 未完成
class Adapter_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.adapters = nn.Sequential(
            Adapter(),
            Adapter(),
            Adapter(),
            Adapter(),
            Adapter(),
            Adapter(),
        ).cuda()
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channel, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.att1.parameters(), self.att2.parameters(), self.att3.parameters(), self.mlp1.parameters(), self.mlp2.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
        # postion
        self.position_encoding_init()
    def position_encoding_init(self):
        max_len = 500
        d_model = 300
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda() # (500, 1 , 300)

################## AttentionLayer ###############
class Attention_Model(nn.Module):
    def __init__(self, learning_rate = 1e-4, layer = 1, head = 3, dropout = 0.1):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=head, dim_feedforward = 50, batch_first = True, dropout = dropout).cuda()
        self.att = nn.TransformerEncoder(encoder_layer, num_layers=layer)
        self.classifier = nn.Sequential(nn.Linear(600, 2)).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.cls = torch.zeros(300)
        self.opter = optim.AdamW(chain(self.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
        # postion
        self.position_encoding_init()
    def position_encoding_init(self):
        max_len = 500
        d_model = 300
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda() # (500, 1 , 300)
    def forward(model, ss, ls):
        vec_cat = model.get_pooled_output(ss, ls) # (600)
        out = model.classifier(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def attention_my_self(self, text):
        vecs = torch.stack([self.cls] + [torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]).cuda() # (?, 300)
        # NOTE: POS
        vecs = vecs.cuda() + self.pe[:vecs.size(0), 0, :] # (?, 300)
        out = self.att(vecs) # (?, 300)
        assert out.size(1) ==300 
        return out
    def get_pooled_output(model, ss, ls):
        left = model.attention_my_self(combine_ss(ss[:2]))[0]  # (?, 300)
        right = model.attention_my_self(combine_ss(ss[2:]))[0]  # (?, 300)
        # Attention
        vec_cat = torch.cat((left, right)) # (600)
        assert len(vec_cat.shape) == 1
        assert vec_cat.size(0) == 600
        return vec_cat

def script():
    model = ModelWrapper(Attention_Model())
    train_save_eval_plot(model, 'ATT_CLS_3layer', batch_size = 32, check_step = 1000, total_step = 50000)

class Attention_Model_Mean(Attention_Model):
    def attention_my_self(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]).cuda() # (?, 300)
        # NOTE: POS
        vecs = vecs.cuda() + self.pe[:vecs.size(0), 0, :] # (?, 300)
        out = self.att(vecs) # (?, 300)
        assert out.size(1) ==300 
        return out
    def get_pooled_output(model, ss, ls):
        left = model.attention_my_self(combine_ss(ss[:2])).mean(0)  # (?, 300)
        right = model.attention_my_self(combine_ss(ss[2:])).mean(0)  # (?, 300)
        # Attention
        vec_cat = torch.cat((left, right)) # (600)
        assert len(vec_cat.shape) == 1
        assert vec_cat.size(0) == 600
        return vec_cat

def script():
    model = ModelWrapper(Attention_Model_Mean(layer = 6, head = 6, dropout = 0))
    train_save_eval_plot(model, 'ATT_6X6_NODROP', batch_size = 32, check_step = 1000, total_step = 50000)

####### 测试norm是否会对loss带来什么影响 ######
def script():
    m = nn.Sequential(
        nn.Linear(5, 3),
        nn.GELU(),
        nn.Linear(3, 2),
    )
    CEL = nn.CrossEntropyLoss()
    inp = torch.rand(1,5)
    out = m(inp) # (1, 2)
    tar = torch.LongTensor([1]) # (1)
    loss = CEL(out, tar)


