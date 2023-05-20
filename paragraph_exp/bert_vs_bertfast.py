from common import save_checkpoint,  cal_prec_rec_f1_v2, Infinite_Dataset, train_save_eval_plot, ModelWrapper, get_test_result, loader
from sector import Sector, Sector_FL
from bert_fasttext import Sector_Fasttext

# NOTE: BERT
def script_bert_only():
    # Models
    model = ModelWrapper(Sector(lr = 2e-5))
    train_save_eval_plot(model, 'bertonly', batch_size = 32, check_step = 500, total_step = 3000)

def script_our():
    # Models
    model = ModelWrapper(Sector_Fasttext(lr = 2e-5))
    train_save_eval_plot(model, 'bert_fasttext', batch_size = 32, check_step = 500, total_step = 3000)

# NOTE: FL
def script_bert_FL():
    for repeat in range(1, 4):
        for FL_RATE in [0.5, 1.0, 2.0, 5.0]:
            # Models
            model = ModelWrapper(Sector_FL(FL_RATE, lr = 2e-5))
            train_save_eval_plot(model, f'fl{repeat}_rate{FL_RATE}', batch_size = 32, check_step = 500, total_step = 3000)


############################ 测试ensemble

@torch.no_grad()
def get_test_result_ensemble(ld, m1, m2):
    # Test
    preds = []
    trues = []
    for idx, (ss, ls) in enumerate(ld):
        out1, _ = m1.forward(ss, ls) # (1,2)
        out2, _ = m2.forward(ss, ls) # (1,2)
        out = torch.cat((out1, out2)).mean(0)
        preds.append(out.argmax().item())
        trues.append(ls[2])
    result = cal_prec_rec_f1_v2(preds, trues)
    return result

ld_test = loader.news.test()
bert = Sector()
PATH = '/home/taku/research/poor-gpt/paragraph_exp/checkpoint/pretrain_bert_fusion/bert_e4_f0.665.checkpoint'
checkpoint = torch.load(PATH)
bert.load_state_dict(checkpoint['model_state_dict'])
test_score_bert = get_test_result(ModelWrapper(bert), ld_test)

fasttext = BILSTM_MEAN()
PATH = '/home/taku/research/poor-gpt/paragraph_exp/checkpoint/pretrain_bert_fusion/BILSTM_MEAN_step9000_f0.589.checkpoint'
checkpoint = torch.load(PATH)
fasttext.load_state_dict(checkpoint['model_state_dict'])
test_score_fasttext = get_test_result(ModelWrapper(fasttext), ld_test)


