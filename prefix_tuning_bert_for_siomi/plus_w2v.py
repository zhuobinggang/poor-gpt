# NOTE 为了提高重用性和trainable_prefix_abstract.py一起开发
from trainable_prefix_abstract import *
from reader import customized_ds
from chart import draw_line_chart
import numpy as np
import torch
from word2vec import get_vec
import itertools

def create_model_for_w2v():
    m = create_model()
    m.amplifier = torch.nn.Linear(300, 768)
    m.opter = torch.optim.AdamW([
        {"params": m.bert.parameters(), 'lr': 1e-5}, 
        {"params": m.amplifier.parameters(), 'lr': 1e-3},
    ])
    return m

def cal_loss(m, item):
    word_org, true_y = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, word_org)
    return loss_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index, true_y)

def dry_run(m, item):
    word_org, true_y = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, word_org)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    word = word.replace(' ', '')
    if word == 'はい':
        return 1, true_y
    elif word == 'いえ':
        return 0, true_y
    else:
        print(f'Turned word {word} to zero!')
        return 0, true_y

def get_inputs_emb_without_pos_info(m, word):
    before_emb = get_emb_without_position_info_for_concat(m, f'[CLS]{word}[SEP]') # (1,?,768)
    fasttext_emb = m.amplifier(torch.from_numpy(get_vec(word))).view(1,1,768).cuda() # (1,1,768)
    after_emb = get_emb_without_position_info_for_concat(m, f'[SEP][MASK][SEP]') # (1,?,768)
    inputs_emb_without_pos_info = torch.cat([before_emb, fasttext_emb, after_emb], dim = 1)
    # after concat
    mask_index = before_emb.shape[1] + 2
    assert mask_index == inputs_emb_without_pos_info.shape[1] - 2
    return inputs_emb_without_pos_info, mask_index


def get_all_vecs(ds):
    wordvecs = []
    for word, label in ds:
        wordvecs.append((word, wordvec(word)))
    return wordvecs

def learning_curve(epochs = 12, batch = 4):
    m = create_model_for_w2v()
    m.bert.requires_grad_(True)
    opter = m.opter
    path = 'learning_curve_with_w2v.png'
    train_ds, test_ds = customized_ds()
    batch_losses = []
    precs = []
    recs = []
    fs = []
    fake_fs = []
    for e in range(epochs):
        batch_losses.append(np.mean(train_one_epoch(m, train_ds, cal_loss, opter, batch = batch)))
        prec, rec, f, fake_f = test(m, test_ds, dry_run, need_random_baseline = True)
        precs.append(prec)
        recs.append(rec)
        fs.append(f)
        fake_fs.append(fake_f)
    x = list(range(epochs))
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [scale_down_batch_losses, precs, recs, fs, fake_fs], ['batch loss', 'precs', 'recs', 'fs', 'random fs'], path = path, colors = ['r','g','b','y', 'k'])
    return m, [scale_down_batch_losses, precs, recs, fs, fake_fs]

