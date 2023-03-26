# 我的重写
# 使用gensim，夕见的w2v模型进行的重写 + 在此之上加上MLP，实现了0.71的精度，证明了神经网络的优越性
# TODO: 需要用fasttext实现同样逻辑
from gensim.models import word2vec
import itertools
from reader import read_regular_ds_zip
import numpy as np
from chart import draw_line_chart
import torch
from torch import nn
from torch import optim


def get_dataset():
    ds = read_regular_ds_zip()
    return tranform_dataset(ds)


def tranform_dataset(ds):
    np.random.seed(seed=10)
    res = []
    for bad, good in ds:
        if np.random.randint(1, 3) == 1:
            res.append(((good, bad), 1))
        else:
            res.append(((bad, good), 2))
    np.random.shuffle(res)
    return res

# 摆正顺序
def reform_ds(ds):
    res = []
    for (w1, w2), label in ds:
        if label == 1:
            res.append((w1, w2))
        elif label == 2:
            res.append((w2, w1))
    assert len(res) == len(ds)
    return res


def run():
    model = word2vec.Word2Vec.load('wikiall2.model')
    ds = get_dataset()
    ds_train = ds[:177]
    ds_test = ds[177:]
    kakudaka_test(model, reform_ds(ds_test), reform_ds(ds_train))


def kakudaka_test(model, ds_test, ds_train):
    yoi = 0
    warui = 0
    for good, bad in ds_test:
        kekka = 0
        for good_learn, bad_learn in ds_train:
            if (good in model.wv) and (bad in model.wv):
                similarity1 = model.wv.similarity(w1=good, w2=good_learn)
                similarity2 = model.wv.similarity(w1=good, w2=bad_learn)
                tmp1 = similarity1 - similarity2
                similarity1 = model.wv.similarity(w1=bad, w2=good_learn)
                similarity2 = model.wv.similarity(w1=bad, w2=bad_learn)
                tmp2 = similarity1 - similarity2
                kekka += tmp1 - tmp2
        if (kekka > 0):
            yoi += 1
        if (kekka < 0):  # and (sw==1):
            warui += 1
    prec = yoi / len(ds_test)
    print(f'PREC: {prec}, ATT: {yoi}, MISS: {warui}')


# =======================  WITH MLP ============================

def train(w2v, train_ds, model, CEL, opter):
    losss = []
    for (w1, w2), label in train_ds:
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(w2v.wv[w1])
        v2 = torch.from_numpy(w2v.wv[w2])
        v_concat = torch.concat((v1, v2)) # 600
        out = model(v_concat).unsqueeze(0) # (1, 2)
        target = torch.LongTensor([label]) # (1)
        loss = CEL(out, target)
        losss.append(loss.item())
        loss.backward()
        opter.step()
        opter.zero_grad()
    epoch_loss = np.mean(losss)
    print(f'Loss: {epoch_loss}')
    return epoch_loss

def test(w2v, model, test_ds):
    attack = 0
    for (w1, w2), label in test_ds:
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(w2v.wv[w1])
        v2 = torch.from_numpy(w2v.wv[w2])
        v_concat = torch.concat((v1, v2)) # 600
        out = model(v_concat) # (2)
        pred = out.argmax().item()
        if pred == label:
            attack += 1
    prec = attack / len(test_ds)
    print(f'Prec: {prec}')
    return prec

def run():
    w2v = word2vec.Word2Vec.load('wikiall2.model')
    model = nn.Sequential(
        nn.Linear(200, 300),
        nn.ReLU(),
        nn.Linear(300, 30),
        nn.ReLU(),
        nn.Linear(30, 2)
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters())
    # DS
    ds = get_dataset()
    ds_train = ds[:177]
    ds_test = ds[177:]
    # 训练
    loss = []
    prec = []
    for e in range(50):
        loss.append(train(w2v, ds_train, model, CEL, opter))
        prec.append(test(w2v, model, ds_test))
    # draw
    draw_line_chart(list(range(50)), [prec, loss], ['precision', 'loss'], path = 'gensim_e50.png', colors = ['red', 'blue'])

