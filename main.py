from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
opter = torch.optim.AdamW(model.parameters(), 1e-5)

def fewshot_dataset():
    inputs = ['man plus power equals [MASK].', 
          'woman plus power equals [MASK].', 
          'sausage plus bread equals [MASK].',
          'man plus wing equals [MASK].',
          'snake plus wing equals [MASK].',
          'light plus dark equals [MASK].',
          'haven plus hell equals [MASK].',
          'animal plus fire equals [MASK].',
          ]
    answers = ['king', 'queen', 'sandwich', 'angel', 'dragon', 'night', 'purgatory', 'phoenix']
    labels = [text.replace('[MASK]', answer) for text, answer in zip(inputs, answers)]
    return inputs, labels

def step(x, y):
    print(x)
    print(y)
    inputs = tokenizer(x, return_tensors="pt")
    labels = tokenizer(y, return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    opter.step()


def finetune_bert():
    inputs, labels = fewshot_dataset()
    for x,y in zip(inputs, labels):
        step(x,y)

quizs = ["man plus ship equals [MASK].", 'bird plus cat equals [MASK].', 'milk plus cat equals [MASK].']

def test(quiz):
    inputs = tokenizer(quiz, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print(tokenizer.decode(predicted_token_id))


