from common import *
from reader import read_data, read_data_full
import numpy as np
import torch

def label2tokenid(label):
    if label == 1:
        return 1015
    elif label == 2:
        return 1016
    elif label == 3:
        return 1017
    elif label == 4:
        return 1018
    elif label == 5:
        return 1019
    else:
        print(f'Wrong label: {label}')
        return None

def word2label(word):
    if word == '1':
        return 1
    elif word == '2':
        return 2
    elif word == '3':
        return 3
    elif word == '4':
        return 4
    elif word == '5':
        return 5
    else:
        print(f'word {word} to -1 !')
        return -1

def concat_ss(ss):
    s_concat = ''
    for s in ss:
        s_concat += f'[MASK]. {s}'
    return s_concat

def cal_loss(m, item):
    ss, ls = item
    tokenizer = m.toker
    s_concat = concat_ss(ss)
    ids = tokenizer(s_concat, return_tensors="pt")["input_ids"] # (1, ?)
    mask_index_list = (ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0] # (?)
    labels = torch.LongTensor([-100] * ids.shape[1])
    for index, label in zip(mask_index_list, ls):
        labels[index] = label2tokenid(label)
    outputs = m.bert(ids.cuda(), labels=labels.unsqueeze(0).cuda())
    return outputs.loss

def dry_run(m, item):
    ss, ls = item
    tokenizer = m.toker
    s_concat = concat_ss(ss)
    ids = tokenizer(s_concat, return_tensors="pt")["input_ids"] # (1, ?)
    mask_index_list = (ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0] # (?)
    with torch.no_grad():
        logits = m.bert(ids.cuda()).logits # (1, ?, 30522)
    mask_logits = logits[0, mask_index_list] # (5, 30522)
    mask_logits = mask_logits.argmax(1) # (5)
    words = [m.toker.decode(token_id) for token_id in mask_logits]
    pred, true =  [word2label(word) for word in words], ls
    return pred, true

def get_predicted_words(m, item):
    ss, ls = item
    tokenizer = m.toker
    s_concat = concat_ss(ss)
    ids = tokenizer(s_concat, return_tensors="pt")["input_ids"] # (1, ?)
    mask_index_list = (ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0] # (?)
    with torch.no_grad():
        logits = m.bert(ids.cuda()).logits # (1, ?, 30522)
    mask_logits = logits[0, mask_index_list] # (5, 30522)
    mask_logits = mask_logits.argmax(1) # (5)
    words = m.toker.decode(mask_logits).split()
    return words



def learning_curve(epochs = 2, batch = 16):
    m = create_model()
    m.bert.requires_grad_(True)
    train_ds_org, test_ds = read_data()
    test_ds = test_ds[:500] # NOTE
    train_ds_splits = [train_ds_org[:1000], train_ds_org[1000:2000], train_ds_org[2000:3000], train_ds_org[3000:4000], train_ds_org[4000:]]
    opter = m.opter
    path = 'prompt_sentence_ordering_curve.png'
    batch_losses = []
    taus = []
    fake_taus = []
    for e in range(epochs):
        for ds in train_ds_splits:
            batch_losses.append(np.mean(train_one_epoch(m, ds, cal_loss, opter, batch = batch, log_inteval = 500)))
            tau, fake_tau = test(m, test_ds, dry_run, need_random_baseline = True)
            taus.append(tau)
            fake_taus.append(fake_tau)
    x = list(range(epochs * len(train_ds_splits)))
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [taus, fake_taus], ['batch loss', 'tau', 'baseline tau'], path = path, colors = ['r','g','k'])
    return m, [scale_down_batch_losses, taus, fake_taus]

def learning_curve_full(epochs = 6, batch = 16):
    m = create_model()
    m.bert.requires_grad_(True)
    train_ds_org, test_ds = read_data_full()
    train_ds_splits = [train_ds_org[:5000], train_ds_org[5000:10000], train_ds_org[10000:15000], train_ds_org[15000:20000], train_ds_org[20000:25000], train_ds_org[25000:30000], train_ds_org[30000:35000], train_ds_org[35000:40000]]
    opter = m.opter
    path = 'prompt_sentence_ordering_curve.png'
    batch_losses = []
    taus = []
    fake_taus = []
    for e in range(epochs):
        for ds in train_ds_splits:
            batch_losses.append(np.mean(train_one_epoch(m, ds, cal_loss, opter, batch = batch, log_inteval = 500)))
            tau, fake_tau = test(m, test_ds, dry_run, need_random_baseline = True)
            taus.append(tau)
            fake_taus.append(fake_tau)
    x = list(range(epochs * len(train_ds_splits)))
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [taus, fake_taus], ['batch loss', 'tau', 'baseline tau'], path = path, colors = ['r','g','k'])
    return m, [scale_down_batch_losses, taus, fake_taus]


