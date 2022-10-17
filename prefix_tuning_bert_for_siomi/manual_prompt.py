from trainable_prefix_abstract import *
from reader import customized_ds


def get_inputs_emb_without_pos_info(m, word):
    emb_before_mask = get_emb_without_position_info_for_concat(m, f'[CLS]「{word}」とは上品な表現ですか？')
    emb_after_mask = get_emb_without_position_info_for_concat(m, f'[MASK]。[SEP]') 
    inputs_emb_without_pos_info = torch.cat([emb_before_mask, emb_after_mask], dim = 1)
    mask_index = emb_before_mask.shape[1]
    return inputs_emb_without_pos_info, mask_index

def cal_loss(m, item):
    word_org, true_y = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, word_org)
    return loss_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index, true_y, positive_token = 'はい', negative_token = 'いえ')

def dry_run(m, item):
    word_org, true_y = item
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, word_org)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    word = word.replace(' ', '')
    if word == 'はい':
        return 1, true_y
    elif word == 'いえ':
        return 0, true_y
    else:
        print(f'Turned word {word} to zero!')
        return 0, true_y

def get_predicted_word(m, text):
    inputs_emb_without_pos_info, mask_index = get_inputs_emb_without_pos_info(m, text)
    word = predict_by_emb_without_position_info(m, inputs_emb_without_pos_info, mask_index)
    return word

def learning_curve(epochs = 12, batch = 4):
    m = create_model()
    m.bert.requires_grad_(True)
    train_ds, test_ds = customized_ds()
    opter = m.opter
    path = 'learning_curve_prompt_tuning_finetune.png'
    batch_losses = []
    precs = []
    recs = []
    fs = []
    fake_fs = []
    for e in range(epochs):
        batch_losses.append(np.mean(train_one_epoch(m, train_ds, cal_loss, opter, batch = batch)))
        prec, rec, f, fake_f = test(m, test_ds, dry_run, need_random_baseline = True)
        precs.append(prec)
        recs.append(rec)
        fs.append(f)
        fake_fs.append(fake_f)
    x = list(range(epochs))
    scale_down_batch_losses = [loss / 10 for loss in batch_losses]
    draw_line_chart(x, [scale_down_batch_losses, precs, recs, fs, fake_fs], ['batch loss', 'precs', 'recs', 'fs', 'random fs'], path = path, colors = ['r','g','b','y', 'k'])
    return m, [scale_down_batch_losses, precs, recs, fs, fake_fs]

def early_stop_5_times(restart = 5, epoch = 5):
    train_ds, test_ds = customized_ds()
    ress = []
    for _ in range(restart):
        m = create_model()
        m.bert.requires_grad_(True)
        opter = m.opter
        for e in range(epoch):
            _ = train_one_epoch(m, train_ds, cal_loss, opter, batch = 4)
        ress.append(test(m, test_ds, dry_run))
    return ress

