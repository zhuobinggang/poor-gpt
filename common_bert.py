from transformers import BertTokenizer, BertForMaskedLM
import torch

def create_them():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.cuda()
    opter = torch.optim.AdamW(model.parameters(), 1e-5)
    return tokenizer, model, opter

