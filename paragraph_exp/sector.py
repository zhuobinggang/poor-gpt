from itertools import chain
import datetime
from transformers import BertJapaneseTokenizer, BertModel
import torch
t = torch
import numpy as np
nn = t.nn
F = t.nn.functional

# 分裂sector, 2vs2的时候，同时判断三个分割点
class Sector(nn.Module):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.lr = lr
        self.bert_size = 768
        self.verbose = False
        self.init_bert()
        self.init_hook()
        self.opter = t.optim.AdamW(
            self.get_should_update(),
            self.lr)
        self.CEL = nn.CrossEntropyLoss()
        self.cuda()
    def init_bert(self):
        self.bert = BertModel.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking')
        self.bert.train()
        self.toker = BertJapaneseTokenizer.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking')
    def get_should_update(self):
        return chain(self.bert.parameters(), self.classifier.parameters())
    def init_hook(self):
        self.classifier = nn.Sequential(  # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
            nn.Linear(self.bert_size, 2),
        )
    def forward(m, ss, ls):
        cls = m.get_pooled_output(ss, ls) # (768)
        logits = m.classifier(cls).unsqueeze(0) # (1, 2)
        targets = torch.LongTensor([ls[2]]).cuda() # (1)
        return logits, targets
    def get_pooled_output(m, ss, ls):
        idxs, sep_idxs = encode_standard(ss, m.toker)
        last_hidden_state = m.bert(idxs.unsqueeze(0).cuda())['last_hidden_state'] # (batch, seq, 768)
        cls = last_hidden_state[0, 0] # (768)
        return cls
    def loss(self, logits, targets):
        return self.CEL(logits, targets)

class Sector_FL(Sector):
    def __init__(self, FL_RATE, lr=2e-5):
        super().__init__(lr)
        self.FL_RATE = FL_RATE
    def loss(self, logits, targets): # (1, 2), (1)
        probs = torch.softmax(logits.squeeze(), 0) # (2)
        y = targets.item()
        pt = probs[1] if y == 1 else probs[0]
        loss = (-1) * torch.log(pt) * torch.pow(1 - pt, self.FL_RATE)
        return loss

# 单个SEP
# NOTE: 处理None
def encode_standard(ss, toker):
    return encode(ss, toker, [False, True, False, False])

# 复数个SEP
# NOTE: 处理NONE
# NOTE: 用一个函数处理所有encode
def encode(ss, toker, seps=[True, True, True, True]):
    PART_LEN_MAX = int(500 / len(ss))  # 默认是512上限，考虑到特殊字符使用500作为分子
    idss = []
    for s in ss:
        ids = [] if s is None else toker.encode(
            s, add_special_tokens=False)  # NOTE: 处理None
        if len(ids) > PART_LEN_MAX:
            # print(f'WARN: len(ids) > PART_LEN_MAX! ===\n{s}')
            idss.append(ids[:PART_LEN_MAX])
        else:
            idss.append(ids)
    combined_ids = [toker.cls_token_id]
    sep_idxs = []
    for index, ids in enumerate(idss):
        if seps[index]:
            ids.append(toker.sep_token_id)
            combined_ids += ids
            sep_idxs.append(len(combined_ids) - 1)
        else:
            combined_ids += ids
    return t.LongTensor(combined_ids), sep_idxs


#################### ADD AT 2023.1.8 ########################
# 为了获取模型的attention以完成修士论文
def view_att_baseline(m, ss):
    # ss, labels = row
    CLS_POS = [0]
    toker = m.toker
    bert = m.bert
    combined_ids, _ = encode_standard(ss, toker)
    attention = bert(
        combined_ids.unsqueeze(0).cuda(),
        output_attentions=True).attentions
    # tuple of (layers = 12, batch = 1, heads = 12, seq_len = 138, seq_len =
    # 138)
    tokens = toker.convert_ids_to_tokens(combined_ids)
    model_view(attention, tokens)


def view_att_my(m, ss):
    # ss, labels = row
    toker = m.toker
    bert = m.bert
    combined_ids, sep_idx = encode(ss, toker)
    attention = bert(
        combined_ids.unsqueeze(0).cuda(),
        output_attentions=True).attentions
    # tuple of (layers = 12, batch = 1, heads = 12, seq_len = 138, seq_len =
    # 138)
    tokens = toker.convert_ids_to_tokens(combined_ids)
    model_view(attention, tokens)

################### 示范script ####################

def script():
    m = Sector()
    ld_train = loader.news.train()
    ss, ls = ld_train[0]
    idxs, sep_idxs = encode_standard(ss, m.toker)
    last_hidden_state = m.bert(idxs.unsqueeze(0).cuda())['last_hidden_state'] # (batch, seq, 768)
    cls = last_hidden_state[0, 0] # (768)
    logits = m.classifier(cls).unsqueeze(0) # (1, 2)
    targets = torch.LongTensor([ls[2]]).cuda() # (1)
    loss = m.CEL(logits, targets)

def script():
    # Models
    model = Sector()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_dev = loader.news.dev()
    # Train
    losses = []
    results = []
    for e in range(5):
        cur_loss = 0
        for idx, (ss, ls) in enumerate(ld_train):
            if idx % 1000 == 0:
                print(f'{idx} / {len(ld_train)}')
            out, tar = model(ss, ls)
            loss = model.CEL(out, tar)
            loss.backward()
            cur_loss += loss.item()
            if idx % 16 == 0:
                losses.append(cur_loss / 16)
                cur_loss = 0
                model.opter.step()
                model.opter.zero_grad()
        model.opter.step()
        model.opter.zero_grad()
        # Test
        result_dev = get_test_result(model, ld_dev)
        results.append(result_dev)
        print(result_dev)
        # Plot loss
        draw_line_chart(range(len(losses)), [losses], ['loss'])
        # save
        save_checkpoint(f'bert_e{e+1}_f{round(result_dev[2], 3)}', model, e, result_dev)


