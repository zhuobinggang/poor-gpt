from common import *
from loader import create_training_data


def cal_loss(m, item):
    model = m.bert
    tokenizer = m.toker
    context, y = item
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    label_text = context.replace('[MASK]', str(y) if y != -1 else '？')
    labels = tokenizer(label_text, return_tensors="pt")["input_ids"]
    labels = torch.where(input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(input_ids = input_ids.cuda(), labels=labels.cuda())
    return outputs.loss

def dry_run(m, item):
    context, y = item
    model = m.bert
    tokenizer = m.toker
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(input_ids = input_ids).logits
    mask_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    if word in ['？', '?']:
        return -1, y
    else:
        try: 
            y_pred = int(word)
            return y_pred, y
        except ValueError as e:
            print(f'ERROR!: model should not output {word}!')
            return -1, y

def get_predicted_word(m, context):
    tokenizer = m.toker
    model = m.bert
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(input_ids = input_ids.cuda()).logits
    mask_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    return word


# m = create_model()

def script(m, epoch = 1):
    losses = []
    ds = create_training_data()
    train_data = ds[:115]
    test_data = ds[115:]
    for e in range(epoch):
        losses += train_one_epoch(m, train_data, cal_loss, m.opter, batch = 4, log_inteval = 4)
    return losses



