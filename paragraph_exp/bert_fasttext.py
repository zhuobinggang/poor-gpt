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


def script():
    # Models
    model = Sector_Fasttext_Pretrained()
    # Datasets
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ld_dev = loader.news.dev()
    # Train
    losses = []
    steps = 1
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
                # eval & save every 100 steps
                steps += 1
                if steps % 100 == 0:
                    # Test
                    result = get_test_result(model, ld_dev)
                    print(result)
                    # Plot loss
                    draw_line_chart(range(len(losses)), [losses], ['loss'])
                    # save
                    save_checkpoint(f'fussion_step{steps}_f{round(result[2], 3)}', model, e, result)
        model.opter.step()
        model.opter.zero_grad()

