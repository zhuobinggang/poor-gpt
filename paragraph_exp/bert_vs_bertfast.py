from common import save_checkpoint, get_test_result, cal_prec_rec_f1_v2, Infinite_Dataset, train_save_eval_plot, ModelWrapper
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
