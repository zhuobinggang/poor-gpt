import torch
import numpy as np
from chart import draw_line_chart
from data_loader import loader

def save_checkpoint(PATH, model, step, score):
    torch.save({
            'model_state_dict': model.m.state_dict(),
            'score': score,
            'step': step
            }, PATH)

def load_checkpoint(PATH):
    raise ValueError('手动load,不要用这个函数')
    model = Sector()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    score = checkpoint['score']
    step = checkpoint['step']
    return model

@torch.no_grad()
def get_test_result(model, ld):
    # Test
    preds = []
    trues = []
    for idx, (ss, ls) in enumerate(ld):
        out, tar = model.forward(ss, ls)
        preds.append(out.argmax().item())
        trues.append(ls[2])
    result = cal_prec_rec_f1_v2(preds, trues)
    return result

def cal_prec_rec_f1_v2(results, targets):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for guess, target in zip(results, targets):
        if guess == 1:
            if target == 1:
                TP += 1
            elif target == 0:
                FP += 1
        elif guess == 0:
            if target == 1:
                FN += 1
            elif target == 0:
                TN += 1
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
    balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
    return prec, rec, f1, balanced_acc


class Infinite_Dataset():
    def __init__(self, ds, shuffle_seed = None):
        self.counter = 0
        if shuffle_seed is None:
            pass
        else:
            np.random.seed(shuffle_seed)
            np.random.shuffle(ds)
        self.ds = ds

    def next(self):
        item = self.ds[self.counter]
        self.counter += 1
        if self.counter >= len(self.ds):
            self.counter = 0
        return item
        

class Loss_Plotter():
    def __init__(self):
        self.loss = []

    def add(self, loss):
        self.loss.append(loss)

    def plot(self, path):
        draw_line_chart(range(len(self.loss)), [self.loss], ['loss'], path = path)

class ModelWrapper():
    def __init__(self, m):
        self.m = m
    def forward(self, ss, ls):
        return self.m(ss,ls)
    def loss(self, out, tar):
        return self.m.loss(out, tar)
    def opter_step(self):
        self.m.opter.step()
        self.m.opter.zero_grad()


# model: ModelWrapper
def train_save_eval_plot(model, save_name, batch_size = 32, check_step = 500, total_step = 5000):
    # Datasets
    ld_train = Infinite_Dataset(loader.news.train(), shuffle_seed = 10)
    ld_dev = loader.news.dev()
    ld_test = loader.news.test()
    # Train
    loser = Loss_Plotter()
    for step in range(total_step):
        batch_loss = []
        for i in range(batch_size):
            ss, ls = ld_train.next()
            out, tar = model.forward(ss, ls)
            loss = model.loss(out, tar)
            loss.backward()
            batch_loss.append(loss.item())
        loser.add(np.mean(batch_loss))
        model.opter_step()
        if (step + 1) % check_step == 0: # Evalue
            result_dev = get_test_result(model, ld_dev)
            result_test = get_test_result(model, ld_test)
            print(f'STEP: {step + 1}\t{result_dev}\t{result_test}')
            # Save
            save_checkpoint(f'checkpoint/{save_name}_step{step + 1}_f{round(result_dev[2], 3)}.checkpoint', model, step+1, {'dev': result_dev, 'test': result_test})
            # Plot
            loser.plot(f'checkpoint/{save_name}_step{step + 1}.png')


def parameters(model):
    return sum(p.numel() for p in model.parameters())

################## Fasttext ###################
def combine_ss(ss):
    res = ''
    for s in ss:
        if s is not None:
            res += s
    return res

############### 梯度消失确认
def check_gradient(m):
    print(m.self_attn.out_proj.weight.grad)


############## Freeze
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False



