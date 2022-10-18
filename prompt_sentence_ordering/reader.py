import json
import numpy as np

def process_ds(annotations):
    length = len(annotations)
    res = []
    for start in range(0, length, 5):
        end = start + 5
        text_in_order = [item['text'] for [item] in annotations[start:end]]
        index = [1,2,3,4,5]
        item = list(zip(text_in_order, index))
        np.random.shuffle(item)
        res.append(([text for text, label in item], [label for text, label in item]))
    return res

def read_data():
    train_json = json.load(open('SIND/val.story-in-sequence.json', 'r'))['annotations']
    test_json = json.load(open('SIND/test.story-in-sequence.json', 'r'))['annotations']
    np.random.seed(seed=2022) # NOTE
    train_ds, test_ds = process_ds(train_json), process_ds(test_json)
    np.random.seed() # NOTE: 恢复random
    return train_ds, test_ds

def read_data_full():
    train_json = json.load(open('SIND/train.story-in-sequence.json', 'r'))['annotations']
    test_json = json.load(open('SIND/test.story-in-sequence.json', 'r'))['annotations']
    np.random.seed(seed=2022) # NOTE
    train_ds, test_ds = process_ds(train_json), process_ds(test_json)
    np.random.seed() # NOTE: 恢复random
    return train_ds, test_ds
