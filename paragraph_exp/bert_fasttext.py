from data_loader import loader
from sector import Sector, encode_standard
from lstm_fasttext import Sector_Traditional_BiLSTM_Attention
from chart import draw_line_chart
from torch import optim
import numpy as np
import torch
from torch import nn
from common import save_checkpoint, get_test_result, cal_prec_rec_f1_v2

##################### Sector + Fasttext ######################
class Sector_Fasttext_Pretrained(nn.Module):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.init_trained_bert_sector()
        self.init_trained_fast_sector()
        self.mlp = nn.Sequential(
            nn.Linear(768 + 1200, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(self.mlp.parameters(), lr = learning_rate)
    def init_trained_bert_sector(self):
        model = Sector()
        checkpoint = torch.load('checkpoint/bert_e4_f0.665.checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.requires_grad_(False) # Freeze
        self.bert_sector = model
    def init_trained_fast_sector(self):
        model = Sector_Traditional_BiLSTM_Attention()
        checkpoint = torch.load('checkpoint/att_e5_f0.568.checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.requires_grad_(False) # Freeze
        self.fast_sector = model
    def forward(self, ss, ls):
        bert_out = self.bert_sector.get_pooled_output(ss, ls) # (768)
        fast_out = self.fast_sector.get_pooled_output(ss, ls) # (1200)
        vec_cat = torch.cat((bert_out.squeeze(), fast_out.squeeze())) # (1200)
        out = self.mlp(vec_cat).unsqueeze(0) # (1, 2)
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(self, out, tar):
        return self.CEL(out, tar)

class Sector_Fasttext(nn.Module):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.init_trained_bert_sector()
        self.init_trained_fast_sector()
        self.mlp = nn.Sequential(
            nn.Linear(768 + 1200, 2),
        ).cuda()
        self.CEL = nn.CrossEntropyLoss()
        self.opter = optim.AdamW(self.parameters(), lr = lr)
    def init_trained_bert_sector(self):
        model = Sector()
        self.bert_sector = model
    def init_trained_fast_sector(self):
        model = Sector_Traditional_BiLSTM_Attention()
        self.fast_sector = model
    def forward(self, ss, ls):
        bert_out = self.bert_sector.get_pooled_output(ss, ls) # (768)
        fast_out = self.fast_sector.get_pooled_output(ss, ls) # (1200)
        vec_cat = torch.cat((bert_out.squeeze(), fast_out.squeeze())) # (1200)
        out = self.mlp(vec_cat).unsqueeze(0) # (1, 2)
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(self, out, tar):
        return self.CEL(out, tar)

