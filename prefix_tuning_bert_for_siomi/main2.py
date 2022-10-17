# NOTE 为了提高重用性和trainable_prefix_abstract.py一起开发
from trainable_prefix_abstract import *
from reader import customized_ds
from chart import draw_line_chart
import numpy as np
import torch

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
    before_prefix_emb = get_emb_without_position_info_for_concat(m, f'[CLS]「{word}」') # (1, ?, 768)
    prefix_emb = m.trainable_prefixs # (1, 10, 768)
    after_prefix_emb = get_emb_without_position_info_for_concat(m, '[MASK]。[SEP]') # (1, 1, 768)
    inputs_emb_without_pos_info = torch.cat([before_prefix_emb, prefix_emb, after_prefix_emb, dim = 1)
    # after concat
    mask_index = before_prefix_emb.shape[1] + prefix_emb.shape[1]
    return inputs_emb_without_pos_info, mask_index

def get_predicted_word(m, text):
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, text)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    return word

# 可以重用
def learning_curve(prompt_tuning_only = True, epochs = 12, batch = 4, lr = 1e-3, prefix_length = 16):
    if prompt_tuning_only:
        m = create_model_with_trainable_prefix(word = '表現', length = prefix_length, learning_rate = lr)
        opter = m.opter_prefixs
        path = 'learning_curve_prompt_tuning_only.png'
    else:
        m = create_model_with_trained_prefix('trained_prefix_len10_epoch10.tch')
        m.trainable_prefixs.requires_grad_(False)
        opter = m.opter
        path = 'learning_curve_prompt_tuning_finetune.png'
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
    return [scale_down_batch_losses, precs, recs, fs, fake_fs]

def redraw():
    pass

def early_stop_5_times(prompt_tuning_only = True, restart = 5, epoch = 10):
    train_ds, test_ds = customized_ds()
    ress = []
    for _ in range(restart):
        if prompt_tuning_only:
            m = create_model_with_trainable_prefix(word = '表現', length = 10, learning_rate = 1e-3)
            opter = m.opter_prefixs
        else:
            m = create_model_with_trained_prefix('trained_prefix_len10_epoch10.tch')
            m.trainable_prefixs.requires_grad_(False)
            opter = m.opter
        for e in range(epoch):
            _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = 4)
        ress.append(test(m, test_ds, dry_run))
    return ress

def train_prefix_and_store():
    train_ds, test_ds = customized_ds()
    m = create_model_with_trainable_prefix(word = '表現', length = 10, learning_rate = 1e-3)
    for e in range(10):
        _ = train_one_epoch(m, train_ds, cal_loss, m.opter_prefixs, batch = 4)
    with torch.no_grad():
        torch.save(m.trainable_prefixs, 'trained_prefix_len10_epoch10.tch')


def train_bert_parameters(m = None):
    if m is None:
        m = create_model_with_trained_prefix('trained_prefix_len10_epoch10.tch')
    train_ds, test_ds = customized_ds()
    m.trainable_prefixs.requires_grad_(False) # freeze prefix
    losses = train_one_epoch(m, train_ds, cal_loss, m.opter, batch = 4)
    print(f'Trained, mean loss: {np.mean(losses)}')
    res = test(m, test_ds, dry_run)
    print(f'TESTED, result: {res}')
    return m
    
    





