# 尝试用fasttext + lstm
import fasttext.util
import torch
from torch import nn
from torch import optim
# Predownload 
# fasttext.util.download_model('ja', if_exists='ignore')
from reader import read_regular_ds_zip
import numpy as np
from numpy import dot
from numpy.linalg import norm
from chart import draw_line_chart
import math

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

def run():
    ft = fasttext.load_model('cc.ja.300.bin')
    w1 = '妊娠'
    w2 = 'おめでた'
    v1 = torch.from_numpy(ft.get_word_vector(w1))
    v2 = torch.from_numpy(ft.get_word_vector(w2))
    model = nn.Sequential(
        nn.Linear(600, 300),
        nn.ReLU(),
        nn.Linear(300, 30),
        nn.ReLU(),
        nn.Linear(30, 2)
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters())
    v_concat = torch.concat((v1, v2)) # 600
    out = model(v_concat) # (2)
    out = out.unsqueeze(0) # (1,2)
    target = torch.LongTensor([1]) # (1)
    loss = CEL(out, target)
    # 训练
    ds = get_dataset()
    train_ds = ds[:177]
    test_ds = ds[177:]
    losss = []
    for (w1, w2), label in train_ds:
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(ft.get_word_vector(w1))
        v2 = torch.from_numpy(ft.get_word_vector(w2))
        v_concat = torch.concat((v1, v2)) # 600
        out = model(v_concat).unsqueeze(0) # (1, 2)
        target = torch.LongTensor([label]) # (1)
        loss = CEL(out, target)
        losss.append(loss.item())
        loss.backward()
        opter.step()
        opter.zero_grad()
    epoch_loss = np.mean(losss)
    # 测试
    attack = 0
    for (w1, w2), label in test_ds:
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(ft.get_word_vector(w1))
        v2 = torch.from_numpy(ft.get_word_vector(w2))
        v_concat = torch.concat((v1, v2)) # 600
        out = model(v_concat) # (2)
        pred = out.argmax().item()
        if pred == label:
            attack += 1
    prec = attack / len(test_ds)

def train(ft, train_ds, model, CEL, opter, batch = 1):
    losss = []
    for idx, ((w1, w2), label) in enumerate(train_ds):
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(ft.get_word_vector(w1))
        v2 = torch.from_numpy(ft.get_word_vector(w2))
        v_concat = torch.concat((v1, v2)) # 600
        out = model(v_concat).unsqueeze(0) # (1, 2)
        target = torch.LongTensor([label]) # (1)
        loss = CEL(out, target)
        losss.append(loss.item())
        loss.backward()
        if (idx + 1) % batch == 0:
            opter.step()
            opter.zero_grad()
    opter.step()
    opter.zero_grad()
    epoch_loss = np.mean(losss)
    print(f'Loss: {epoch_loss}')
    return epoch_loss

def test(ft, model, test_ds):
    attack = 0
    for (w1, w2), label in test_ds:
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(ft.get_word_vector(w1))
        v2 = torch.from_numpy(ft.get_word_vector(w2))
        v_concat = torch.concat((v1, v2)) # 600
        out = model(v_concat) # (2)
        pred = out.argmax().item()
        if pred == label:
            attack += 1
    prec = attack / len(test_ds)
    print(f'Prec: {prec}')
    return prec

def run_batch(train_ds = None, test_ds = None):
    ft = fasttext.load_model('cc.ja.300.bin')
    model = nn.Sequential(
        nn.Linear(600, 300),
        nn.ReLU(),
        nn.Linear(300, 30),
        nn.ReLU(),
        nn.Linear(30, 2)
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters())
    if train_ds is None:
        ds = get_dataset()
        train_ds = ds[:177]
        test_ds = ds[177:]
    # 训练
    loss = []
    prec = []
    for e in range(50):
        loss.append(train(ft, train_ds, model, CEL, opter, batch = 16))
        prec.append(test(ft, model, test_ds))
    # draw
    draw_line_chart(list(range(50)), [prec, loss], ['precision', 'loss'], path = 'fasttext_e50.png', colors = ['red', 'blue'])


def load_aug_dataset():
    f = open('generated_100_words_better2.txt', 'r')
    lines = f.readlines()
    f.close()
    bad_goods = []
    for line in lines:
        line = line.strip()
        bad, good = line.split('->')
        bad = bad.strip()
        good = good.strip()
        bad_goods.append((bad, good))
    return tranform_dataset(bad_goods)

def shuffle_with_dataset_aug(ds1, ds2):
    ds_new = ds1.copy() + ds2.copy()
    np.random.seed(10)
    np.random.shuffle(ds_new)
    return ds_new

def run_batch_aug():
    ds = get_dataset()
    train_ds = ds[:177]
    test_ds = ds[177:]
    ds_aug = load_aug_dataset()[:200]
    ds_train_combined = shuffle_with_dataset_aug(ds_aug, train_ds)
    run_batch(ds_train_combined, test_ds)

def run_siomi_algorithm():
    ds = get_dataset()
    train_ds = ds[:177]
    test_ds = ds[177:]
    good_bads = []
    for (w1,w2),label in train_ds:
        if label == 1:
            good_bads.append((w1, w2))
        elif label == 2:
            good_bads.append((w2, w1))
        else:
            print('???')
    goods = [good for good, bad in good_bads]
    bads = [bad for good, bad in good_bads]
    ft = fasttext.load_model('cc.ja.300.bin')
    good_vecs = [ft.get_word_vector(good) for good in goods]
    bad_vecs = [ft.get_word_vector(bad) for bad in bads]
    good_mean = np.mean(good_vecs, 0)
    bad_mean = np.mean(bad_vecs, 0)
    # Test
    preds = []
    trues = []
    for (w1, w2), label in test_ds:
        trues.append(label)
        v1, v2 = ft.get_word_vector(w1), ft.get_word_vector(w2)
        dist1 = np.linalg.norm(v1 - good_mean) - np.linalg.norm(v1 - bad_mean)
        dist2 = np.linalg.norm(v2 - good_mean) - np.linalg.norm(v2 - bad_mean)
        if dist1 > dist2:
            preds.append(1)
        else:
            preds.append(2)
    attack = 0
    for pred, true in zip(preds, trues):
        if pred == true:
            attack += 1
    prec = attack / len(test_ds)
    

# ================ 使用fasttext实现夕见的算法 =================

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

def similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    if math.isnan(cos_sim):
        return -1
    else:
        return cos_sim

# 夕见算法
def kakudaka_test(ft, ds_test, ds_train):
    yoi = 0
    warui = 0
    for good, bad in ds_test:
        v_good = ft.get_word_vector(good)
        v_bad = ft.get_word_vector(bad)
        kekka = 0
        for good_learn, bad_learn in ds_train:
            v_good_learn = ft.get_word_vector(good_learn)
            v_bad_learn = ft.get_word_vector(bad_learn)
            similarity1 = similarity(v_good, v_good_learn)
            similarity2 = similarity(v_good, v_bad_learn)
            tmp1 = similarity1 - similarity2
            similarity1 = similarity(v_bad, v_good_learn)
            similarity2 = similarity(v_bad, v_bad_learn)
            tmp2 = similarity1 - similarity2
            kekka += tmp1 - tmp2
        if (kekka > 0):
            yoi += 1
        elif (kekka < 0):  # and (sw==1):
            warui += 1
        else:
            print(f'???? {good} {bad}, {kekka}')
            warui += 1
    prec = yoi / len(ds_test)
    print(f'PREC: {prec}, ATT: {yoi}, MISS: {warui}')

def run_siomi():
    ft = fasttext.load_model('cc.ja.300.bin')
    ds = get_dataset()
    train_ds = ds[:177]
    test_ds = ds[177:]
    kakudaka_test(ft, reform_ds(test_ds), reform_ds(train_ds))
    # 数据增强……
    ds_aug = load_aug_dataset()[:200]
    ds_train_combined = shuffle_with_dataset_aug(ds_aug, train_ds)
    kakudaka_test(ft, reform_ds(test_ds), reform_ds(ds_train_combined))


# =============== 试试用w1 - w2来直接分类，理论上神经网络应该不会低于算法才对 ======
# =============== 结论： 性能确实提高了，但是也只有0.8，直接算是0.85 ===========

def distance_classify():
    ft = fasttext.load_model('cc.ja.300.bin')
    model = nn.Sequential(
        nn.Linear(300, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
        nn.Sigmoid(),
    )
    model = nn.Sequential(
        nn.Linear(300, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(model.parameters(), lr = 1e-3)
    ds = get_dataset()
    train_ds = ds[:177]
    test_ds = ds[177:]
    ds_aug = load_aug_dataset()[:200]
    ds_train_combined = shuffle_with_dataset_aug(ds_aug, train_ds)
    # ds_train_combined = shuffle_with_dataset_aug([], train_ds)
    # 训练
    loss = []
    prec = []
    for e in range(50):
        loss.append(train_point_wise(ft, ds_train_combined, model, CEL, opter, batch = 16))
        prec.append(test_point_wise(ft, model, test_ds))
    # draw
    draw_line_chart(list(range(50)), [prec, loss], ['precision', 'loss'], path = 'fasttext_e50.png', colors = ['red', 'blue'])

def train_point_wise(ft, train_ds, model, CEL, opter, batch = 1):
    losss = []
    for idx, ((w1, w2), label) in enumerate(train_ds):
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(ft.get_word_vector(w1))
        v2 = torch.from_numpy(ft.get_word_vector(w2))
        out = model(v1 - v2).unsqueeze(0) # (1, 2)
        target = torch.LongTensor([label]) # (1)
        loss = CEL(out, target)
        losss.append(loss.item())
        loss.backward()
        if (idx + 1) % batch == 0:
            opter.step()
            opter.zero_grad()
    opter.step()
    opter.zero_grad()
    epoch_loss = np.mean(losss)
    print(f'Loss: {epoch_loss}')
    return epoch_loss

def test_point_wise(ft, model, test_ds):
    attack = 0
    for (w1, w2), label in test_ds:
        label = label - 1 # 2 -> 1, 1 -> 0
        v1 = torch.from_numpy(ft.get_word_vector(w1))
        v2 = torch.from_numpy(ft.get_word_vector(w2))
        out = model(v1 - v2) # (2)
        pred = out.argmax().item()
        if pred == label:
            attack += 1
    prec = attack / len(test_ds)
    print(f'Prec: {prec}')
    return prec
