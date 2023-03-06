from main import *
from sklearn.metrics import f1_score
import time

SEED = 999
np.random.seed(SEED)
print(f'SET RANDOM SEED = {SEED}')

def dataset_test(shuffle = False, skip_negative = False):
    train_ds = []
    test_ds = []
    for name in ['goutou', 'smartphone', 'siro']:
        train_ds += create_training_data(name = name)
    for name in ['jisin', 'camera', 'dinasour']:
        test_ds += create_training_data(name = name)
    if shuffle:
        np.random.shuffle(train_ds)
        np.random.shuffle(test_ds)
    if skip_negative:
        train_ds = [(ss, label) for ss, label in train_ds if label != -1]
        test_ds = [(ss, label) for ss, label in test_ds if label != -1]
    return train_ds, test_ds

def try_train(m, epoch = 1, ds = None):
    losses = []
    if ds is None:
        ds, _ = dataset_test(shuffle = True, skip_negative = True)
    for e in range(epoch):
        losses += train_one_epoch(m, ds, cal_loss, m.opter, batch = 16, log_inteval = 16)
    return losses

def test_skip(m, tds = None, skip = True):
    if tds is None:
        _, tds = dataset_test(shuffle = False)
    preds = []
    trues = []
    for item in tds:
        pred, true = dry_run(m, item)
        preds.append(pred)
        trues.append(true)
    # cal prec
    zero_ones = []
    for pred, true in zip(preds, trues):
        if (not skip) or (skip and true != -1):
            if pred == true:
                zero_ones.append(1)
            else:
                zero_ones.append(0)
    return sum(zero_ones) / len(zero_ones)

# Batch Running
def script(epoch = 20):
    start_time = time.time()
    _, tds = dataset_test(shuffle = False, skip_negative = False)
    # skip_negative
    m = create_model()
    precs_skip = []
    ds, _ = dataset_test(shuffle = True, skip_negative = True)
    for i in range(epoch):
        _ = train_one_epoch(m, ds, cal_loss, m.opter, batch = 4, log_inteval = 16)
        dd = round(test_skip(m, tds), 4)
        print(dd)
        precs_skip.append(dd)
        save_model(m, f'skip_negative_e{i}_{dd}')
    # Do not skip negative
    m = create_model()
    precs_no_skip = []
    ds, _ = dataset_test(shuffle = True, skip_negative = False)
    for i in range(epoch):
        _ = train_one_epoch(m, ds, cal_loss, m.opter, batch = 4, log_inteval = 16)
        dd = round(test_skip(m, tds), 4)
        print(dd)
        precs_no_skip.append(dd)
        save_model(m, f'with_negative_e{i}_{dd}')
    print("--- %s seconds ---" % (time.time() - start_time))
    return precs_skip, precs_no_skip



