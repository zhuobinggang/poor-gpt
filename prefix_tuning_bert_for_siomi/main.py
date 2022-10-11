from common import create_model
from reader import customized_ds, calculate_result
import torch
import numpy as np
from chart import draw_line_chart

def create_model_with_trainable_prefix():
    m = create_model()
    # Initialize prefix embeddings and opter
    prefix_ids = m.toker.encode('とは堅い表現ですか？', add_special_tokens = False) 
    prefix_ids = torch.LongTensor([prefix_ids]).cuda() # (8)
    with torch.no_grad():
        prefix_embs = m.bert.bert.embeddings(prefix_ids) # (1, 8 , 768)
        trainable_prefixs = prefix_embs.clone() # (1, 8, 768)
    trainable_prefixs.requires_grad_(True)
    # Assign
    m.trainable_prefixs = trainable_prefixs
    m.opter_prefixs = torch.optim.AdamW([trainable_prefixs], 1e-3)
    m.trainable_prefixs_length = trainable_prefixs.shape[1]
    return m

def create_model_with_trained_prefix():
    m = create_model()
    # Initialize prefix embeddings and opter
    prefix_embs = torch.load('trained_prefix0.tch') # (1, 8 , 768)
    prefix_embs.requires_grad_(True)
    # Assign
    m.trainable_prefixs = prefix_embs
    m.opter_prefixs = torch.optim.AdamW([prefix_embs], 1e-3)
    m.trainable_prefixs_length = prefix_embs.shape[1]
    return m

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
    total_length = m.trainable_prefixs_length + ids_length_without_prefix
    return prediction_scores, mask_token_index_updated, total_length


def cal_loss(model_with_trainable_prefix, text, label):
    m = model_with_trainable_prefix
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



def train_one_epoch(m, batch = 8):
    train_ds, test_ds = customized_ds()
    train_ds_length = len(train_ds)
    batch_losses = []
    batch_loss = 0
    for index, (text, label) in enumerate(train_ds):
        loss = cal_loss(m, text, label)
        loss.backward()
        batch_loss += loss.item()
        if (index + 1) % batch == 0:
            m.opter_prefixs.step()
            m.opter_prefixs.zero_grad()
            batch_losses.append(batch_loss)
            batch_loss = 0
            print(f'{index + 1} / {train_ds_length}')
    m.opter_prefixs.step()
    m.opter_prefixs.zero_grad()
    return batch_losses

def predicted_token_to_labels(predicted_tokens, turn_word_to_zero = False):
    y_pred = []
    for token in predicted_tokens:
        token = token.replace(' ', '')
        if token == 'はい':
            y_pred.append(1)
        elif token == 'いえ':
            y_pred.append(0)
        else:
            if turn_word_to_zero:
                print(f'Turned word {token} to zero!')
                y_pred.append(0)
            else:
                y_pred.append(token)
    return y_pred

def get_predicted_word(m, text):
    prediction_scores, mask_token_index_updated, _ = insert_prefix_and_get_prediction_scores(m, text)
    predicted_token_id = prediction_scores[0, mask_token_index_updated].argmax().item()
    return m.toker.decode(predicted_token_id)

def test(m):
    train_ds, test_ds = customized_ds()
    predicted_tokens = []
    y_true = []
    for index, (text, label) in enumerate(test_ds):
        predicted_tokens.append(get_predicted_word(m, text))
        y_true.append(label)
    # y_pred = [1 if token.replace(' ', '') == 'はい' else 0 for token in predicted_tokens]
    return predicted_tokens, y_true

def script_record():
    m = create_model()
    # Initialize prefix embeddings and opter
    prefix_ids = m.toker.encode('とは堅い表現ですか？', add_special_tokens = False) 
    prefix_ids = torch.LongTensor([prefix_ids]).cuda() # (8)
    with torch.no_grad():
        prefix_embs = m.bert.bert.embeddings(prefix_ids) # (1, 8 , 768)
        trainable_prefixs = prefix_embs.clone() # (1, 8, 768)
    trainable_prefixs.requires_grad_(True)
    trainable_prefixs_length = trainable_prefixs.shape[1]
    opter = torch.optim.AdamW([trainable_prefixs], 1e-3)
    # Load data
    train_ds, test_ds = customized_ds()
    text, label = test_ds[0]
    # one step
    pattern_incomplete = f'[CLS]「{text}」[MASK]。[SEP]'
    ids = m.toker.encode(pattern_incomplete, add_special_tokens = False)
    ids = torch.LongTensor([ids]).cuda() 
    ids_length_without_prefix = ids.shape[1]
    mask_token_index = (ids == m.toker.mask_token_id)[0].nonzero(as_tuple=True)[0].item() # 放到mask token前面
    embedding_outputs = m.bert.bert.embeddings(ids) # (1, ?, 768)
    encoder_inputs = torch.cat([embedding_outputs[:,:mask_token_index], trainable_prefixs[:,:], embedding_outputs[:,mask_token_index:]], 1) # (1, ? + 8, 768)
    encoder_outputs = m.bert.bert.encoder(encoder_inputs)
    sequence_output = encoder_outputs[0] # (1, ? + 8, 768)
    prediction_scores = m.bert.cls(sequence_output) # (1, ? + 8, 32000)
    # calculate loss
    # 1. 计算はい和いえ的id
    # 2. 将target id放进mask_token_index_updated的位置，其他都是-100
    mask_token_index_updated = mask_token_index + trainable_prefixs_length
    total_length = trainable_prefixs_length + ids_length_without_prefix
    labels = [-100] * total_length
    target_label_word = 'はい' if label == 1 else 'いえ'
    target_token_id = m.toker.encode(target_label_word, add_special_tokens = False)[0]
    labels[mask_token_index_updated] = target_token_id
    labels = torch.LongTensor(labels)
    loss_fct = torch.nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(prediction_scores.view(-1, m.bert.config.vocab_size), labels.view(-1))

def run():
    m = create_model_with_trainable_prefix()
    batch_losses = []
    for e in range(10):
        batch_losses += train_one_epoch(m)
    predicted_tokens, y_true = test(m)
    y_pred = predicted_token_to_labels(predicted_tokens)
    print(y_pred)
    print(y_true)


# ================= Finetune =====================

def continue_train_bert_paramaters_customized_trainds(train_ds, m, batch = 8):
    train_ds_length = len(train_ds)
    batch_losses = []
    batch_loss = 0
    for index, (text, label) in enumerate(train_ds):
        loss = cal_loss(m, text, label)
        loss.backward()
        batch_loss += loss.item()
        if (index + 1) % batch == 0:
            m.opter.step()
            m.opter.zero_grad()
            batch_losses.append(batch_loss)
            batch_loss = 0
            print(f'{index + 1} / {train_ds_length}')
    m.opter.step()
    m.opter.zero_grad()
    return batch_losses

def get_curve(model_with_trained_prefix):
    m = model_with_trained_prefix
    train_ds, _ = customized_ds()
    fs = []
    for start in range(0, len(train_ds), 16):
        end = start + 16
        _ = continue_train_bert_paramaters_customized_trainds(train_ds[start:end], m, 8)
        predicted_tokens, y_true = test(m)
        y_pred = predicted_token_to_labels(predicted_tokens, turn_word_to_zero = True)
        prec, rec, f = calculate_result(y_pred, y_true)
        if f > np.max(fs):
            print(f'new record at {start}: {prec}, {rec}, {f}')
        fs.append(f)
    return fs

def continue_train_bert_paramaters(m, batch = 8):
    train_ds, _ = customized_ds()
    return continue_train_bert_paramaters_customized_trainds(train_ds)


def draw_fss(fss):
    xs = list(range(len(fss)))
    draw_line_chart(xs, [fss], ['trined prefix result'])


