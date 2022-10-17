import types
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt


def create_model(learning_rate = 1e-5):
    res = types.SimpleNamespace()
    # NOTE
    res.bert = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.opter = torch.optim.AdamW(res.bert.parameters(), learning_rate)
    res.bert.cuda()
    res.bert.train()
    return res

# REUSABLE
def create_model_with_trainable_prefix(word, length, learning_rate = 5e-5):
    m = create_model()
    # Initialize prefix embeddings and opter
    token_to_fill_id = m.toker.encode(word, add_special_tokens = False)[0]
    prefix_ids = torch.LongTensor([token_to_fill_id] * length).unsqueeze(0).cuda() # (8)
    with torch.no_grad():
        prefix_embs = m.bert.bert.embeddings(prefix_ids) # (1, 8 , 768)
        trainable_prefixs = prefix_embs.clone() # (1, 8, 768)
    trainable_prefixs.requires_grad_(True)
    # Assign
    m.trainable_prefixs = trainable_prefixs
    m.opter_prefixs = torch.optim.AdamW([trainable_prefixs], learning_rate)
    m.trainable_prefixs_length = trainable_prefixs.shape[1]
    return m

# REUSABLE
def create_model_with_trained_prefix(path):
    m = create_model()
    # Initialize prefix embeddings and opter
    prefix_embs = torch.load(path) # (1, 8 , 768)
    print(f'Loaded prefix embeddings with shape {prefix_embs.shape}!')
    prefix_embs.requires_grad_(True)
    # Assign
    m.trainable_prefixs = prefix_embs
    m.opter_prefixs = torch.optim.AdamW([prefix_embs], 1e-3)
    m.trainable_prefixs_length = prefix_embs.shape[1]
    return m

# REUSABLE, 抽象函数
def train_one_epoch(m, train_ds, cal_loss, opter, batch = 4, log_inteval = 4):
    train_ds_length = len(train_ds)
    batch_losses = []
    batch_loss = 0
    for index, item in enumerate(train_ds):
        loss = cal_loss(m, item)
        loss.backward()
        batch_loss += loss.item()
        if (index + 1) % log_inteval == 0:
            print(f'{index + 1} / {train_ds_length}')
        if (index + 1) % batch == 0:
            opter.step()
            opter.zero_grad()
            m.bert.zero_grad()
            batch_loss = batch_loss / batch # Mean
            batch_losses.append(batch_loss)
            batch_loss = 0
    opter.step()
    opter.zero_grad()
    m.bert.zero_grad()
    return batch_losses

# REUSABLE, 抽象函数
def test(m, test_ds, dry_run, need_random_baseline = False):
    y_preds = []
    y_trues = []
    for index, item in enumerate(test_ds):
        pred, true = dry_run(m, item)
        y_preds.append(pred)
        y_trues.append(true)
    if not need_random_baseline:
        return calculate_result(y_preds, y_trues)
    else:
        prec, rec, f = calculate_result(y_preds, y_trues)
        random_y_preds = np.random.randint(0, 2, len(y_trues))
        _, _, fake_f = calculate_result(random_y_preds, y_trues)
        return prec, rec, f, fake_f


# ABSTRACT
def cal_loss(m, item):
    pass

# ABSTRACT
def dry_run(m, item):
    pred = 0
    true = 1
    return pred, true

# REUSABLE
def calculate_result(pred_y, true_y):
    f = f1_score(true_y, pred_y, average='macro')
    prec = precision_score(true_y, pred_y, average='macro')
    rec = recall_score(true_y, pred_y, average='macro')
    print(f'F: {f}, PRECISION: {prec}, RECALL: {rec}')
    return prec, rec, f


# ============= SET ==================
# NOTE: 关于这部分代码谷歌 Huggingface BertEmbeddings Source Code
def get_emb_without_position_info_for_concat(m, s):
    ids = m.toker.encode(s, add_special_tokens = False)
    ids = torch.LongTensor([ids]).cuda()
    emb_without_position_info = m.bert.bert.embeddings.word_embeddings(ids)
    return emb_without_position_info

def embedding_with_position_info(m, emb_without_position_info):
    return m.bert.bert.embeddings(inputs_embeds = emb_without_position_info)

def bert_encoder(m, emb_with_position_info):
    return m.bert.bert.encoder(emb_with_position_info)

def predict_by_emb_without_position_info(m, inputs_emb, mask_index):
    inputs_emb_with_position_info = embedding_with_position_info(m, inputs_emb)
    encoder_outputs = bert_encoder(m, inputs_emb_with_position_info)
    sequence_output = encoder_outputs[0]
    prediction_scores = m.bert.cls(sequence_output) # (1, total_length, 32000)
    predicted_token_id = prediction_scores[0, mask_index].argmax().item()
    return m.toker.decode(predicted_token_id)

def loss_by_emb_without_position_info(m, inputs_emb, mask_index, label, positive_token = 'はい', negative_token = 'いえ'):
    inputs_emb_with_position_info = embedding_with_position_info(m, inputs_emb)
    encoder_outputs = bert_encoder(m, inputs_emb_with_position_info)
    sequence_output = encoder_outputs[0]
    prediction_scores = m.bert.cls(sequence_output) # (1, total_length, 32000)
    # cal loss
    total_length = prediction_scores.shape[1]
    labels = [-100] * total_length
    target_label_word = positive_token if label == 1 else negative_token
    target_token_id = m.toker.encode(target_label_word, add_special_tokens = False)[0]
    labels[mask_index] = target_token_id
    labels = torch.LongTensor(labels).cuda()
    loss_fct = torch.nn.CrossEntropyLoss()
    masked_lm_loss = loss_fct(prediction_scores.view(-1, m.bert.config.vocab_size), labels.view(-1))
    return masked_lm_loss
# ============= END SET ==================

def draw_line_chart(x, ys, legends, path = 'dd.png', colors = None):
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors is not None:
            plt.plot(x[:len(y)], y, label = l, color = colors[i])
        else:
            plt.plot(x[:len(y)], y, label = l)
    plt.legend()
    plt.savefig(path)
