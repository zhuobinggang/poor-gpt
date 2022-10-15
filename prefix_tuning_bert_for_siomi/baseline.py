from reader import customized_ds
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from trainable_prefix_abstract import train_one_epoch, test
from chart import draw_line_chart
import torch
import datetime
import numpy as np
import types

def create_model(learning_rate = 1e-5):
    res = types.SimpleNamespace()
    # NOTE
    res.bert = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.opter = torch.optim.AdamW(res.bert.parameters(), learning_rate)
    res.bert.cuda()
    res.bert.train()
    return res

def cal_loss(m, item):
    word, label = item
    inputs = m.toker(word, return_tensors="pt", truncation=True)["input_ids"]
    labels = torch.LongTensor([label])
    loss = m.bert(inputs.cuda(), labels=labels.cuda()).loss
    return loss

def dry_run(m, item):
    word, label = item
    inputs = m.toker(word, return_tensors="pt", truncation=True)["input_ids"]
    with torch.no_grad():
        logits = m.bert(inputs.cuda()).logits
    pred = logits.argmax().item()
    true = label
    return pred, true

def learning_curve(epochs = 12, batch = 4):
    m = create_model()
    opter = m.opter
    train_ds, test_ds = customized_ds()
    path = 'baseline_curve.png'
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

def early_stop_5_times(m, restart = 5, epoch = 5):
    train_ds, test_ds = customized_ds()
    ress = []
    for _ in range(restart):
        m = create_model()
        opter = m.opter
        for e in range(epoch):
            _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = 4)
        ress.append(test(m, test_ds, dry_run))
    return ress


