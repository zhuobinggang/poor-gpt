from bert import *

def load_aug_dataset():
    f = open('generated_100_words_better2.txt', 'r')
    lines = f.readlines()
    f.close()
    bad_goods = []
    for line in lines:
        line = line.strip()
        bad, good = line.split('->')
        bad = bad.strip()
        good = good.strip()
        bad_goods.append((bad, good))
    return tranform_dataset(bad_goods)

def shuffle_with_dataset_aug(ds1, ds2):
    ds_new = ds1.copy() + ds2.copy()
    np.random.seed(10)
    np.random.shuffle(ds_new)
    return ds_new


def save_model(m, name):
    torch.save(m, f'./check_points/{name}.tch')

def run():
    # baseline
    m = create_model()
    tokenizer = m.toker
    model = m.bert
    ds_base = get_dataset()
    ds_aug = load_aug_dataset() # 100 -> 200
    ds_base_train = ds_base[:177] # 177
    ds_base_test = ds_base[177:]
    ds_train_combined = shuffle_with_dataset_aug(ds_base_train, [])
    # train and draw
    precs = []
    fake_precs = []
    xs = list(range(50))
    for e in xs:
        lss = train_one_epoch(m, ds_train_combined, cal_loss, m.opter, batch = 8, log_inteval = 170)
        prec, fake_prec = test(m, ds_base_test, dry_run, True)
        precs.append(prec)
        fake_precs.append(fake_prec)
        print(f'epoch {e} : {prec}')
        if (e + 1) % 5 == 0:
            save_model(model, f'bert_e{e + 1}_prec{round(prec, 3)}.tch')
    # 数据增强
    m = create_model()
    tokenizer = m.toker
    model = m.bert
    ds_base = get_dataset()
    ds_aug = load_aug_dataset() # 100 -> 200
    ds_base_train = ds_base[:177] # 177
    ds_base_test = ds_base[177:]
    ds_train_combined = shuffle_with_dataset_aug(ds_base_train, ds_aug)
    # train and draw
    precs = []
    fake_precs = []
    xs = list(range(50))
    for e in xs:
        lss = train_one_epoch(m, ds_train_combined, cal_loss, m.opter, batch = 8, log_inteval = 170)
        prec, fake_prec = test(m, ds_base_test, dry_run, True)
        precs.append(prec)
        fake_precs.append(fake_prec)
        print(f'epoch {e} : {prec}')
        if (e + 1) % 5 == 0:
            save_model(model, f'bert_aug_e{e + 1}_prec{round(prec, 3)}.tch')



