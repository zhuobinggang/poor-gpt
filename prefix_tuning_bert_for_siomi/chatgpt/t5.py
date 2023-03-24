# 为了证明生成模型在单词单位上的理解力高于MLM模型进行此实验
# 如果成功，继续进行数据增强实验
# 如果失败，……看看能不能用chatGPT来写论文吧
from transformers import T5Tokenizer, T5ForConditionalGeneration
from reader import read_regular_ds_zip
import numpy as np
import torch

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
    tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)
    model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
    model = model.cuda()
    input_ids = tokenizer("単語「育ち」よりも詩的な表現は: ", return_tensors="pt").input_ids.cuda()
    opter = torch.optim.AdamW(model.parameters(), 2e-5)
    ds = get_dataset()
    # train
    ds_train = ds[:177]
    for idx, ((word1, word2), label) in enumerate(ds_train):
        if (idx+1) % 16 == 0:
            print(f'{idx + 1}/{len(ds_train)}')
        input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
        labels = tokenizer(f"{label}", return_tensors="pt").input_ids.cuda()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        opter.step()
        opter.zero_grad()
    # test
    ds_test = ds[177:]
    preds = []
    for idx, ((word1, word2), label) in enumerate(ds_test):
        if (idx+1) % 16 == 0:
            print(f'{idx + 1}/{len(ds_test)}')
        input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids)
        preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # batch
    ds_train = ds[:177]
    ds_test = ds[177:]
    predss = []
    for e in range(5):
        for idx, ((word1, word2), label) in enumerate(ds_train):
            if (idx+1) % 16 == 0:
                print(f'{idx + 1}/{len(ds_train)}')
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{label}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            opter.step()
            opter.zero_grad()
        preds = []
        for idx, ((word1, word2), label) in enumerate(ds_test):
            if (idx+1) % 16 == 0:
                print(f'{idx + 1}/{len(ds_test)}')
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
            outputs = model.generate(input_ids)
            preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        predss.append(preds)
    # calculate
    count = 0
    trues = [label for _, label in ds_test]
    for pred, true in zip(predss[4], trues):
        if int(pred) == true:
            count += 1
    # generate & classify, 同时生成+分类
    ds_train = ds[:177]
    ds_test = ds[177:]
    predss = []
    for e in range(50):
        for idx, ((word1, word2), label) in enumerate(ds_train):
            if (idx+1) % 16 == 0:
                print(f'{idx + 1}/{len(ds_train)}')
            # generate
            (good, bad) = (word1, word2) if label == 1 else (word2, word1)
            input_ids = tokenizer(f"{bad} より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{good}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            # classify
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現の番号は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{label}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            # step
            opter.step()
            opter.zero_grad()
        # test
        preds = []
        for idx, ((word1, word2), label) in enumerate(ds_test):
            if (idx+1) % 16 == 0:
                print(f'{idx + 1}/{len(ds_test)}')
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現の番号は: ", return_tensors="pt").input_ids.cuda()
            outputs = model.generate(input_ids)
            preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        predss.append(preds)
    # cal
    count = 0
    trues = [label for _, label in ds_test]
    for pred, true in zip(preds, trues):
        if pred == str(true):
            count += 1


def run2():
    tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)
    model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
    model = model.cuda()
    opter = torch.optim.AdamW(model.parameters(), 2e-5)
    ds = get_dataset()
    ds_train = ds[:177]
    ds_test = ds[177:]
    # batch run
    for e in range(50):
        for idx, ((word1, word2), label) in enumerate(ds_train):
            # generate
            (good, bad) = (word1, word2) if label == 1 else (word2, word1)
            input_ids = tokenizer(f"{bad} より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{good}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            # classify
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現の番号は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{label}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            # step
            opter.step()
            opter.zero_grad()
        # test
        preds = []
        for idx, ((word1, word2), label) in enumerate(ds_test):
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現の番号は: ", return_tensors="pt").input_ids.cuda()
            outputs = model.generate(input_ids)
            preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # cal
        count = 0
        trues = [label for _, label in ds_test]
        for pred, true in zip(preds, trues):
            if pred == str(true):
                count += 1
        prec = count / 100
        print(f'prec: {prec}')
        # save
        if (e+1) % 5 == 0:
            print(f'SAVE MODEL EPOCH {e+1}')
            save_model(model, f't5_e{e + 1}_prec{round(prec, 3)}.tch')


def save_model(m, name):
    torch.save(m, f'./check_points/{name}.tch')

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

# 数据增强
def run3():
    tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese", is_fast=True)
    model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
    model = model.cuda()
    opter = torch.optim.AdamW(model.parameters(), 2e-5)
    ds = get_dataset()
    ds_train = ds[:177]
    ds_test = ds[177:]
    ds_aug = load_aug_dataset()
    ds_train_combined = shuffle_with_dataset_aug(ds_aug, ds_train)
    # batch run
    for e in range(50):
        for idx, ((word1, word2), label) in enumerate(ds_train_combined):
            # generate
            (good, bad) = (word1, word2) if label == 1 else (word2, word1)
            input_ids = tokenizer(f"{bad} より詩的な表現は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{good}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            # classify
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現の番号は: ", return_tensors="pt").input_ids.cuda()
            labels = tokenizer(f"{label}", return_tensors="pt").input_ids.cuda()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            # step
            opter.step()
            opter.zero_grad()
        # test
        preds = []
        for idx, ((word1, word2), label) in enumerate(ds_test):
            input_ids = tokenizer(f"1. {word1} 2. {word2} | より詩的な表現の番号は: ", return_tensors="pt").input_ids.cuda()
            outputs = model.generate(input_ids)
            preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        # cal
        count = 0
        trues = [label for _, label in ds_test]
        for pred, true in zip(preds, trues):
            if pred == str(true):
                count += 1
        prec = count / 100
        print(f'prec: {prec}')
        # save
        if (e+1) % 5 == 0:
            print(f'SAVE MODEL EPOCH {e+1}')
            save_model(model, f't5_e{e + 1}_prec{round(prec, 3)}.tch')

