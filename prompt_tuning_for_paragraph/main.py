import types
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import torch

def create_model(learning_rate = 1e-5):
    res = types.SimpleNamespace()
    # NOTE
    res.bert = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.opter = torch.optim.AdamW(res.bert.parameters(), learning_rate)
    res.bert.cuda()
    res.bert.train()
    return res


