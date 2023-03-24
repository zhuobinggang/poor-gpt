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

def run():
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
    xs = list(range(10))
    for _ in xs:
        lss = train_one_epoch(m, ds_train_combined, cal_loss, m.opter, batch = 8, log_inteval = 16)
        prec, fake_prec = test(m, ds_base_test, dry_run, True)
        precs.append(prec)
        fake_precs.append(fake_prec)
    draw_line_chart(xs, [precs, fake_precs], ['precs', 'random baseline'], path = 'aug_300.png', colors = ['red', 'gray'])


