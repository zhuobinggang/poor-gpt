from reader import read_regular_ds_zip
# from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from trainable_prefix_abstract import create_model, train_one_epoch, test
from chart import draw_line_chart
import torch
import datetime
import numpy as np
import types

# shuffle，每次都相同，可以重复实验
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

def prompt(word1, word2):
    return f" 1.{word1} 2.{word2} | [MASK]番目の単語は詩的です。"

def dry_run(m, item):
    tokenizer = m.toker
    model = m.bert
    (word1, word2), true = item
    inputs = tokenizer(prompt(word1, word2), return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'].cuda()).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    try:
        pred = int(word)
    except ValueError as e:
        print(f'将单词{word}变换成数字出错，设置为3')
        pred = 3
    return pred, true

def cal_loss(m, item):
    tokenizer = m.toker
    model = m.bert
    (word1, word2), true = item
    inputs = tokenizer(prompt(word1, word2), return_tensors="pt")
    # create label
    labels = tokenizer(prompt(word1, word2), return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.mask_token_id] = tokenizer.encode(str(true),add_special_tokens = False)[0]
    # mask labels of non-[MASK] tokens
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(input_ids=inputs['input_ids'].cuda(), labels=labels.cuda())
    return outputs.loss


def cal_prec(preds, trues):
    prec = sum([1 if left == right else 0 for left, right in zip(preds, trues)]) / len(trues)
    return prec

def test(m, test_ds, dry_run, need_random_baseline = False):
    preds = []
    trues = []
    for index, item in enumerate(test_ds):
        pred, true = dry_run(m, item)
        preds.append(pred)
        trues.append(true)
    prec = cal_prec(preds, trues)
    if not need_random_baseline:
        return prec
    else:
        random_y_preds = np.random.randint(1, 3, len(trues))
        fake_prec = cal_prec(random_y_preds, trues)
        return prec, fake_prec

def run():
    m = create_model()
    tokenizer = m.toker
    model = m.bert
    ds = get_dataset()
    train_ds = ds[:177]
    test_ds = ds[177:]
    # train and draw
    precs = []
    fake_precs = []
    xs = list(range(10))
    for _ in xs:
        lss = train_one_epoch(m, train_ds, cal_loss, m.opter, batch = 4, log_inteval = 4)
        prec, fake_prec = test(m, test_ds, dry_run, True)
        precs.append(prec)
        fake_precs.append(fake_prec)
    draw_line_chart(xs, [precs, fake_precs], ['precs', 'random baseline'], path = 'dd.png', colors = ['red', 'gray'])



