from data_loader import loader
import fasttext
import torch
from torch import nn
from torch import optim
import numpy as np
from chart import draw_line_chart
from fugashi import Tagger
from itertools import chain

# 使用fasttext来整段落分割
def get_wordvec_script(word):
    ft = fasttext.load_model('cc.ja.300.bin')
    ft.get_word_vector(word)

def combine_ss(ss):
    res = ''
    for s in ss:
        if s is not None:
            res += s
    return res


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
