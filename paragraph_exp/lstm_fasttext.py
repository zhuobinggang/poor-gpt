from data_loader import loader
import fasttext
import torch
from torch import nn
from torch import optim
from torch.utils.data import Sampler, Dataset
import numpy as np
from chart import draw_line_chart
from fugashi import Tagger
from itertools import chain
from common import combine_ss, ModelWrapper, train_save_eval_plot, parameters

def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    ld_train = loader.news.train()
    ss, ls = ld_train[0]
    left = combine_ss(ss[:2])
    right = combine_ss(ss[2:])
    left_vec = ft.get_word_vector(left) # (300)
    right_vec = ft.get_word_vector(right) # (300)
    # 神经网络
    model = nn.Sequential(
        nn.Linear(600, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters())
    # 训练
    v1 = torch.from_numpy(left_vec)
    v2 = torch.from_numpy(left_vec)
    v_cat = torch.cat(v1, v2) # (600)
    out = model(v_cat).unsqueeze(0) # (1, 2)
    target = torch.LongTensor([ls[2]]) # (1)
    loss = CEL(out, target)



def cal_prec_rec_f1_v2(results, targets):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  for guess, target in zip(results, targets):
    if guess == 1:
      if target == 1:
        TP += 1
      elif target == 0:
        FP += 1
    elif guess == 0:
      if target == 1:
        FN += 1
      elif target == 0:
        TN += 1
  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  rec = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
  balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
  balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
  return prec, rec, f1, balanced_acc

def concat(ss, ls, ft, model):
    left = combine_ss(ss[:2])
    right = combine_ss(ss[2:])
    left_vec = ft.get_word_vector(left) # (300)
    right_vec = ft.get_word_vector(right) # (300)
    # 训练
    v1 = torch.from_numpy(left_vec)
    v2 = torch.from_numpy(right_vec)
    v_cat = torch.cat((v1, v2)) # (600)
    out = model(v_cat).unsqueeze(0) # (1, 2)
    # loss & step
    target = torch.LongTensor([ls[2]]) # (1)
    return out, target

# Fasttext, MLP, concat => 全部输出0，好像没有在学习的样子？先shuffle一下试试，然后batch一下，然后打印loss
def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    ld_train = loader.news.train()
    # 神经网络
    model = nn.Sequential(
        nn.Linear(600, 300),
        nn.ReLU(),
        nn.Linear(300, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters(), lr = 1e-3)
    losses = []
    # Train
    for idx, (ss, ls) in enumerate(ld_train):
        if idx % 1000 == 0:
            print(f'{idx} / {len(ld_train)}')
        out, target = concat(ss, ls, ft, model)
        loss = CEL(out, target)
        loss.backward()
        losses.append(loss.item())
        opter.step()
        opter.zero_grad()
    # Test
    ld_test = loader.news.test()
    preds = []
    trues = []
    for idx, (ss, ls) in enumerate(ld_test):
        out, target = concat(ss, ls, ft, model)
        preds.append(out.argmax().item())
        trues.append(ls[2])
    print(cal_prec_rec_f1_v2(preds, trues))


# Fasttext, MLP, concat => 全部输出0，好像没有在学习的样子？
# 先shuffle一下试试 => 一样，没有用
# 然后batch一下，然后打印loss => 都一样，没有用
def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    # 神经网络
    model = nn.Sequential(
        nn.Linear(600, 300),
        nn.ReLU(),
        nn.Linear(300, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    # Train
    for idx, (ss, ls) in enumerate(ld_train):
        if idx % 1000 == 0:
            print(f'{idx} / {len(ld_train)}')
        out, target = concat(ss, ls, ft, model)
        loss = CEL(out, target)
        loss.backward()
        losses.append(loss.item())
        if idx % 16 == 0:
            opter.step()
            opter.zero_grad()
    opter.step()
    opter.zero_grad()
    # Test
    ld_test = loader.news.test()
    preds = []
    trues = []
    for idx, (ss, ls) in enumerate(ld_test):
        out, target = concat(ss, ls, ft, model)
        preds.append(out.argmax().item())
        trues.append(ls[2])
    print(cal_prec_rec_f1_v2(preds, trues))
    # Plot loss
    draw_line_chart(range(len(losses)), [losses], ['loss'])


def substract(ss, ls, ft, model):
    left = combine_ss(ss[:2])
    right = combine_ss(ss[2:])
    left_vec = ft.get_word_vector(left) # (300)
    right_vec = ft.get_word_vector(right) # (300)
    # 训练
    v1 = torch.from_numpy(left_vec)
    v2 = torch.from_numpy(right_vec)
    v_sub = v1 - v2 # (600)
    out = model(v_sub).unsqueeze(0) # (1, 2)
    # loss & step
    target = torch.LongTensor([ls[2]]) # (1)
    return out, target

# 减法
def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    # 神经网络
    model = nn.Sequential(
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    # Train
    for idx, (ss, ls) in enumerate(ld_train):
        if idx % 1000 == 0:
            print(f'{idx} / {len(ld_train)}')
        out, target = substract(ss, ls, ft, model)
        loss = CEL(out, target)
        loss.backward()
        losses.append(loss.item())
        if idx % 16 == 0:
            opter.step()
            opter.zero_grad()
    opter.step()
    opter.zero_grad()
    # Test
    ld_test = loader.news.test()
    preds = []
    trues = []
    for idx, (ss, ls) in enumerate(ld_test):
        out, target = substract(ss, ls, ft, model)
        preds.append(out.argmax().item())
        trues.append(ls[2])
    print(cal_prec_rec_f1_v2(preds, trues))
    # Plot loss
    draw_line_chart(range(len(losses)), [losses], ['loss'])

# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# 手稿
def script():
    rnn = nn.LSTM(300, 300, 1, batch_first = True)
    model = nn.Sequential(
        nn.Linear(600, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(chain(rnn.parameters(), model.parameters()), lr = 1e-3)
    tagger = Tagger('-Owakati')
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ss,ls = ld_train[0]
    left = combine_ss(ss[:2])
    right = combine_ss(ss[2:])
    left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
    right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
    _, (left_vec, _) = rnn(left_vecs.unsqueeze(0)) # (1, 1, 300)
    _, (right_vec, _) = rnn(right_vecs.unsqueeze(0)) # (1, 1, 300)
    vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze()))
    out = model(vec_cat).unsqueeze(0) # (1, 2)
    # loss & step
    tar = torch.LongTensor([ls[2]]) # (1)
    loss = CEL(out, tar)

# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# 正式测试
def script():
    # Models
    model = Sector_Traditional()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_test = loader.news.test()
    # Train
    losses = []
    cur_loss = 0
    for idx, (ss, ls) in enumerate(ld_train):
        if idx % 1000 == 0:
            print(f'{idx} / {len(ld_train)}')
        out, tar = model(ss, ls)
        loss = model.CEL(out, tar)
        loss.backward()
        cur_loss += loss.item()
        if idx % 16 == 0:
            losses.append(cur_loss / 16)
            cur_loss = 0
            opter.step()
            opter.zero_grad()
    opter.step()
    opter.zero_grad()
    # Test
    preds = []
    trues = []
    for idx, (ss, ls) in enumerate(ld_test):
        out, tar = model(ss, ls)
        preds.append(out.argmax().item())
        trues.append(ls[2])
    print(cal_prec_rec_f1_v2(preds, trues))
    # Plot loss
    draw_line_chart(range(len(losses)), [losses], ['loss'])


class Sector_Traditional(nn.Module):
    def __init__(self, learning_rate = 1e-3):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(600, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.rnn.parameters(), self.mlp.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        rnn = model.rnn
        mlp = model.mlp
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        _, (left_vec, _) = rnn(left_vecs.unsqueeze(0).cuda()) # (1, 1, 300)
        _, (right_vec, _) = rnn(right_vecs.unsqueeze(0).cuda()) # (1, 1, 300)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze()))
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar

# 减法
class Sector_Traditional_Sub(nn.Module):
    def __init__(self, learning_rate = 1e-3):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(300, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.rnn.parameters(), self.mlp.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        rnn = model.rnn
        mlp = model.mlp
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        _, (left_vec, _) = rnn(left_vecs.unsqueeze(0).cuda()) # (1, 1, 300)
        _, (right_vec, _) = rnn(right_vecs.unsqueeze(0).cuda()) # (1, 1, 300)
        vec_sub = left_vec - right_vec
        out = mlp(vec_sub.squeeze()).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar



# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# NOTE: 减法
# 正式测试
def script():
    # Models
    model = Sector_Traditional_Sub()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_test = loader.news.test()
    # Train
    losses = []
    results = []
    for e in range(10):
        cur_loss = 0
        for idx, (ss, ls) in enumerate(ld_train):
            if idx % 1000 == 0:
                print(f'{idx} / {len(ld_train)}')
            out, tar = model(ss, ls)
            loss = model.CEL(out, tar)
            loss.backward()
            cur_loss += loss.item()
            if idx % 16 == 0:
                losses.append(cur_loss / 16)
                cur_loss = 0
                model.opter.step()
                model.opter.zero_grad()
        model.opter.step()
        model.opter.zero_grad()
        # Test
        preds = []
        trues = []
        for idx, (ss, ls) in enumerate(ld_test):
            out, tar = model(ss, ls)
            preds.append(out.argmax().item())
            trues.append(ls[2])
        result = cal_prec_rec_f1_v2(preds, trues)
        results.append(result)
        print(result)
        # Plot loss
        draw_line_chart(range(len(losses)), [losses], ['loss'])


# 双方向
class Sector_Traditional_BiLSTM(nn.Module):
    def __init__(self, learning_rate = 1e-3):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(1200, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.rnn.parameters(), self.mlp.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        rnn = model.rnn
        mlp = model.mlp
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        left_vecs, (_, _) = rnn(left_vecs.cuda()) # (?, 600)
        right_vecs, (_, _) = rnn(right_vecs.cuda()) # (?, 600)
        left_vec = left_vecs[-1] # (600)
        right_vec = right_vecs[0] # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar

# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# NOTE: 双方向
# 正式测试
def script():
    # Models
    model = Sector_Traditional_BiLSTM()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_test = loader.news.test()
    # Train
    losses = []
    results = []
    for e in range(10):
        cur_loss = 0
        for idx, (ss, ls) in enumerate(ld_train):
            if idx % 1000 == 0:
                print(f'{idx} / {len(ld_train)}')
            out, tar = model(ss, ls)
            loss = model.CEL(out, tar)
            loss.backward()
            cur_loss += loss.item()
            if idx % 16 == 0:
                losses.append(cur_loss / 16)
                cur_loss = 0
                model.opter.step()
                model.opter.zero_grad()
        model.opter.step()
        model.opter.zero_grad()
        # Test
        preds = []
        trues = []
        for idx, (ss, ls) in enumerate(ld_test):
            out, tar = model(ss, ls)
            preds.append(out.argmax().item())
            trues.append(ls[2])
        result = cal_prec_rec_f1_v2(preds, trues)
        results.append(result)
        print(result)
        # Plot loss
        draw_line_chart(range(len(losses)), [losses], ['loss'])


# 双方向 & Max Pooling
class Sector_Traditional_BiLSTM_Max(nn.Module):
    def __init__(self, learning_rate = 1e-3):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(1200, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.rnn.parameters(), self.mlp.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        rnn = model.rnn
        mlp = model.mlp
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        left_vecs, (_, _) = rnn(left_vecs.cuda()) # (?, 600)
        right_vecs, (_, _) = rnn(right_vecs.cuda()) # (?, 600)
        with torch.no_grad():
            assert len(left_vecs.shape) == 2 
            assert left_vecs.shape[1] == 600
            left_max_index = left_vecs.sum(1).argmax()
            right_max_index = right_vecs.sum(1).argmax()
        left_vec = left_vecs[left_max_index] # (600)
        right_vec = right_vecs[right_max_index] # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar

# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# NOTE: 双方向 & MAX
# 正式测试
def script():
    # Models
    model = Sector_Traditional_BiLSTM_Max()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_test = loader.news.test()
    # Train
    losses = []
    results = []
    for e in range(10):
        cur_loss = 0
        for idx, (ss, ls) in enumerate(ld_train):
            if idx % 1000 == 0:
                print(f'{idx} / {len(ld_train)}')
            out, tar = model(ss, ls)
            loss = model.CEL(out, tar)
            loss.backward()
            cur_loss += loss.item()
            if idx % 16 == 0:
                losses.append(cur_loss / 16)
                cur_loss = 0
                model.opter.step()
                model.opter.zero_grad()
        model.opter.step()
        model.opter.zero_grad()
        # Test
        preds = []
        trues = []
        for idx, (ss, ls) in enumerate(ld_test):
            out, tar = model(ss, ls)
            preds.append(out.argmax().item())
            trues.append(ls[2])
        result = cal_prec_rec_f1_v2(preds, trues)
        results.append(result)
        print(result)
        # Plot loss
        draw_line_chart(range(len(losses)), [losses], ['loss'])


# =============== 双方向 + attention ==================
def save_checkpoint(name, model, epoch, val_prec):
    PATH = f'checkpoint/{name}.checkpoint'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_prec': val_prec,
            }, PATH)

def load_checkpoint(PATH):
    model = Sector_Traditional_BiLSTM_Attention()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_prec = checkpoint['val_prec']
    return model



# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# NOTE: 双方向 & attention
# 正式测试
def script_attention():
    # Models
    model = Sector_Traditional_BiLSTM_Attention()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_dev = loader.news.dev()
    # Train
    losses = []
    results = []
    for e in range(10):
        cur_loss = 0
        for idx, (ss, ls) in enumerate(ld_train):
            if idx % 1000 == 0:
                print(f'{idx} / {len(ld_train)}')
            out, tar = model(ss, ls)
            loss = model.CEL(out, tar)
            loss.backward()
            cur_loss += loss.item()
            if idx % 16 == 0:
                losses.append(cur_loss / 16)
                cur_loss = 0
                model.opter.step()
                model.opter.zero_grad()
        model.opter.step()
        model.opter.zero_grad()
        # Test
        preds = []
        trues = []
        for idx, (ss, ls) in enumerate(ld_dev):
            out, tar = model(ss, ls)
            preds.append(out.argmax().item())
            trues.append(ls[2])
        result = cal_prec_rec_f1_v2(preds, trues)
        results.append(result)
        print(result)
        # Plot loss
        draw_line_chart(range(len(losses)), [losses], ['loss'])
        # save
        save_checkpoint(f'att_e{e+1}_f{round(result[2], 3)}', model, e, result)

# =============== 双方向 + 全注意 ==================

# 双方向 & Attention
class Sector_Traditional_BiLSTM_All(nn.Module):
    def __init__(self, learning_rate = 1e-3):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(1200, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(chain(self.rnn.parameters(), self.mlp.parameters()), lr = learning_rate)
        self.tagger = Tagger('-Owakati')
    def forward(model, ss, ls):
        rnn = model.rnn
        mlp = model.mlp
        tagger = model.tagger
        ft = model.ft
        left = combine_ss(ss[:2])
        right = combine_ss(ss[2:])
        left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
        right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
        left_index = left_vecs.shape[0] - 1
        right_index = left_vecs.shape[0]
        vecs = torch.cat((left_vecs, right_vecs)) # (?, 300)
        outs, (_, _) = rnn(vecs.unsqueeze(0).cuda()) # (1, ?, 600)
        outs = outs.squeeze()
        assert len(outs.shape) == 2
        assert outs.shape[1] == 600
        # Attention
        left_vec = outs[left_index] # (600)
        right_vec = outs[right_index] # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar


	

# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# NOTE: 双方向 & attention
# 正式测试
def script():
    # Models
    model = Sector_Traditional_BiLSTM_All()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_dev = loader.news.test()
    # Train
    losses = []
    results = []
    for e in range(10):
        cur_loss = 0
        for idx, (ss, ls) in enumerate(ld_train):
            if idx % 1000 == 0:
                print(f'{idx} / {len(ld_train)}')
            out, tar = model(ss, ls)
            loss = model.CEL(out, tar)
            loss.backward()
            cur_loss += loss.item()
            if idx % 16 == 0:
                losses.append(cur_loss / 16)
                cur_loss = 0
                model.opter.step()
                model.opter.zero_grad()
        model.opter.step()
        model.opter.zero_grad()
        # validate
        preds = []
        trues = []
        for idx, (ss, ls) in enumerate(ld_dev):
            out, tar = model(ss, ls)
            preds.append(out.argmax().item())
            trues.append(ls[2])
        result = cal_prec_rec_f1_v2(preds, trues)
        results.append(result)
        print(result)
        # Plot loss
        draw_line_chart(range(len(losses)), [losses], ['loss'])


############## 降低学习率重新实验: 比较LSTM和Attention的性能
class Adapter(nn.Module):
    def __init__(self, in_channel, hidden_channen):
        super().__init__()
        self.down = nn.Linear(in_channel, hidden_channen)
        self.gelu = nn.GELU()
        self.up = nn.Linear(hidden_channen, in_channel)
    def forward(self, x):
        h = self.down(x) 
        h = self.gelu(h)
        h = self.up(h)
        return x + h

# 双方向 & Attention
class BILSTM_ATT_MEAN(nn.Module):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.tagger = Tagger('-Owakati')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True)
        self.attention = nn.Sequential(
            nn.Linear(600, 1),
            nn.Softmax(dim = 0),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1200, 2),
        )
        self.adapter = Adapter(600, 100)
        self.CEL = nn.CrossEntropyLoss()
        self.cuda()
        self.opter = optim.AdamW(chain(self.parameters()), lr = learning_rate)
    def forward(model, ss, ls):
        mlp = model.mlp
        vec_cat = model.get_pooled_output(ss, ls) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def create_vecs(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]) # (?, 300)
        vecs, (_, _) = self.rnn(vecs.cuda()) # (?, 600)
        # NOTE: ATT
        vecs = self.attention(vecs) * self.adapter(vecs)
        return vecs
    def get_pooled_output(self, ss, ls):
        left_vecs = self.create_vecs(combine_ss(ss[:2]))
        right_vecs = self.create_vecs(combine_ss(ss[2:]))
        assert len(left_vecs.shape) == 2 
        assert left_vecs.shape[1] == 600
        # Mean
        left_vec = left_vecs.mean(dim = 0) # (600)
        right_vec = right_vecs.mean(dim = 0) # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        return vec_cat


def script():
    model = ModelWrapper(BILSTM_ATT_MEAN())
    train_save_eval_plot(model, 'BILSTM_ATT_MEAN', batch_size = 32, check_step = 1000, total_step = 50000)

# 不要Adapter?
class BILSTM_ATT_MEAN2(nn.Module):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.tagger = Tagger('-Owakati')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True)
        self.attention = nn.Sequential(
            nn.Linear(600, 1),
            nn.Softmax(dim = 0),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1200, 2),
        )
        self.adapter = Adapter(600, 100)
        self.CEL = nn.CrossEntropyLoss()
        self.cuda()
        self.opter = optim.AdamW(chain(self.parameters()), lr = learning_rate)
    def forward(model, ss, ls):
        mlp = model.mlp
        vec_cat = model.get_pooled_output(ss, ls) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def create_vecs(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]) # (?, 300)
        vecs, (_, _) = self.rnn(vecs.cuda()) # (?, 600)
        # NOTE: ATT
        vecs = self.attention(vecs) * vecs
        return vecs
    def get_pooled_output(self, ss, ls):
        left_vecs = self.create_vecs(combine_ss(ss[:2]))
        right_vecs = self.create_vecs(combine_ss(ss[2:]))
        assert len(left_vecs.shape) == 2 
        assert left_vecs.shape[1] == 600
        # Mean
        left_vec = left_vecs.mean(dim = 0) # (600)
        right_vec = right_vecs.mean(dim = 0) # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        return vec_cat


# NOTE: NO Attention
class BILSTM_MEAN(BILSTM_ATT_MEAN): 
    def create_vecs(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]) # (?, 300)
        vecs, (_, _) = self.rnn(vecs.cuda()) # (?, 600)
        return vecs

def script():
    # check att
    model = ModelWrapper(BILSTM_MEAN())
    m = model.m
    ld = loader.news.train()
    ss, ls = ld[99]
    out, tar = m(ss,ls)
    loss = m.loss(out, tar)
    loss.backward()
    check_gradient(m)
    train_save_eval_plot(model, 'BILSTM_MEAN', batch_size = 32, check_step = 1000, total_step = 50000)


