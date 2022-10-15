# NOTE 为了提高重用性和trainable_prefix_abstract.py一起开发
from trainable_prefix_abstract import create_model_with_trainable_prefix, train_one_epoch, test, create_model_with_trained_prefix
from reader import customized_ds
from chart import draw_line_chart
import numpy as np
import torch

# 大部分逻辑来自huggingface文档
def cal_loss(m, item):
    text, label = item
    prediction_scores, mask_token_index_updated, total_length = insert_prefix_and_get_prediction_scores(m, text)
    # calculate loss
    # 1. 计算はい和いえ的id
    # 2. 将target id放进mask_token_index_updated的位置，其他都是-100
    labels = [-100] * total_length
    target_label_word = 'はい' if label == 1 else 'いえ'
    target_token_id = m.toker.encode(target_label_word, add_special_tokens = False)[0]
    labels[mask_token_index_updated] = target_token_id
    labels = torch.LongTensor(labels).cuda()
    loss_fct = torch.nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(prediction_scores.view(-1, m.bert.config.vocab_size), labels.view(-1))
    return masked_lm_loss

def word2label(word):
    word = word.replace(' ', '')
    if word == 'はい':
        return 1
    elif word == 'いえ':
        return 0
    else:
        print(f'Turned word {word} to zero!')
        return 0


def dry_run(m, item):
    text, label = item
    prediction_scores, mask_token_index_updated, _ = insert_prefix_and_get_prediction_scores(m, text)
    predicted_token_id = prediction_scores[0, mask_token_index_updated].argmax().item()
    word = m.toker.decode(predicted_token_id)
    y_pred = word2label(word)
    return y_pred, label


# 大部分逻辑来自huggingface文档
def insert_prefix_and_get_prediction_scores(model_with_trainable_prefix, text):
    m = model_with_trainable_prefix
    pattern_incomplete = f'[CLS]「{text}」[MASK]。[SEP]'
    ids = m.toker.encode(pattern_incomplete, add_special_tokens = False)
    ids = torch.LongTensor([ids]).cuda() 
    ids_length_without_prefix = ids.shape[1]
    #NOTE: The position to place prefix is right before [MASK]
    mask_token_index = (ids == m.toker.mask_token_id)[0].nonzero(as_tuple=True)[0].item() 
    embedding_outputs = m.bert.bert.embeddings(ids) # (1, ids_length_without_prefix, 768)
    encoder_inputs = torch.cat([embedding_outputs[:,:mask_token_index], m.trainable_prefixs[:,:], embedding_outputs[:,mask_token_index:]], 1) # (1, ids_length_without_prefix + trainable_prefixs_length, 768)
    encoder_outputs = m.bert.bert.encoder(encoder_inputs)
    sequence_output = encoder_outputs[0] # (1, ids_length_without_prefix + trainable_prefixs_length, 768)
    prediction_scores = m.bert.cls(sequence_output) # (1, ids_length_without_prefix + trainable_prefixs_length, 32000)
    mask_token_index_updated = mask_token_index + m.trainable_prefixs_length
    assert mask_token_index_updated == ids_length_without_prefix + m.trainable_prefixs_length - 3 # NOTE: 倒数第三个
    total_length = m.trainable_prefixs_length + ids_length_without_prefix
    return prediction_scores, mask_token_index_updated, total_length

def get_predicted_word(m, text):
    prediction_scores, mask_token_index_updated, _ = insert_prefix_and_get_prediction_scores(m, text)
    predicted_token_id = prediction_scores[0, mask_token_index_updated].argmax().item()
    return m.toker.decode(predicted_token_id)

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
    
    





