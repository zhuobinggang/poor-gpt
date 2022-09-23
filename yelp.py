from datasets import load_dataset
from common_bert import create_them
import numpy as np
dataset = load_dataset("yelp_review_full")
import torch
# tokenizer, model, opter = create_them()

train = dataset['train']
test = dataset['test']

def patterned(text):
    return f'It was [MASK]. {text}'

def pattern_and_label(item, number = 1):
    if number == 1:
        label = item['label'] # number
        text = item['text'] # string
        pattern = f'It was [MASK]. {text}'
        verbalized_pattern = pattern.replace('[MASK]', verbalize(label))
        return pattern, verbalized_pattern
    else:
        pass

def verbalize(label):
    vs = ['terrible', 'bad', 'okay', 'good', 'great']
    return vs[label]

def unverbalize(word):
    dic = {
        'terrible': 0,
        'bad': 1,
        'okay': 2,
        'good': 3,
        'great': 4
    }
    if word in dic:
        return dic[word]
    else:
        return word

def random_choice_fewshot_training_set(size = 32):
    low = 0
    high = 650000
    array = np.random.randint(low, high, size = size)
    results = []
    for index in array:
        results.append(train[int(index)])
    return results

def create_inputs_and_labels_from_fewshot_set(fewshot_set):
    patterns = []
    verbalized_patterns = []
    for item in fewshot_set:
        pattern, verbalized_pattern = pattern_and_label(item)
        patterns.append(pattern)
        verbalized_patterns.append(verbalized_pattern)
    return patterns, verbalized_patterns

def step(x, y, tokenizer, model, opter):
    print(x)
    print(y)
    inputs = tokenizer(x, return_tensors="pt", truncation=True)
    labels = tokenizer(y, return_tensors="pt", truncation=True)["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    print(loss.item())
    loss.backward()
    opter.step()


# tokenizer, model, opter = create_them()

def run(tokenizer, model, opter):
    fewshot = random_choice_fewshot_training_set(size = 32)
    patterns, verbalized_patterns = create_inputs_and_labels_from_fewshot_set(fewshot)
    for x,y in zip(patterns, verbalized_patterns):
        step(x, y, tokenizer, model, opter)

def get_predicted_word(tokenizer, model, text):
    inputs = tokenizer(patterned(text), return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    return word

def get_test_result(tokenizer, model):
    results = []
    # for index, item in enumerate(test):
    for index in range(3000):
        if index % 100 == 0:
            # print(f'{index}/50000')
            print(f'{index}/3000')
        item = test[index]
        inputs = tokenizer(patterned(item['text']), return_tensors="pt", truncation=True)['input_ids']
        inputs = inputs.cuda()
        with torch.no_grad():
            logits = model(inputs).logits
        mask_token_index = (inputs == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        result = tokenizer.decode(predicted_token_id)
        results.append(result)
    return results

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

