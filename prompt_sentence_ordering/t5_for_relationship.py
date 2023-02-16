# T5用于从故事中提取关系
from datasets import load_dataset
dataset = load_dataset("docred")
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
import numpy as np
import time
import types


def create_model(learning_rate = 2e-5, size = 'small'):
    res = types.SimpleNamespace()
    # NOTE
    # res.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    # res.toker = T5Tokenizer.from_pretrained("google/flan-t5-small")
    res.t5 = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{size}")
    res.toker = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}")
    # res.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    # res.toker = T5Tokenizer.from_pretrained("google/flan-t5-base")
    res.opter = torch.optim.AdamW(res.t5.parameters(), learning_rate)
    res.t5.cuda()
    res.t5.train()
    return res

# DocRED

def sents2doc(sents):
    return ' '.join([' '.join(tokens) for tokens in sents])


def flatten(sents):
    res = []
    for tokens in sents:
        res += tokens
    return res

def print_nice(lst):
    for item in lst:
        print(item)


# Get available entities
def vertexset2output(vertexset):
    # return ';'.join([','.join([item['name'] for item in cluster]) for cluster in vertexset])
    return ';'.join([f'[{idx}]' + ','.join([item['name'] for item in cluster]) for idx, cluster in enumerate(vertexset)])

def label2triple(labels, vertexset):
    entities = [cluster[0]['name'] for idx, cluster in enumerate(vertexset)]
    heads = labels['head']
    tails = labels['tail']
    relation_texts = labels['relation_text']
    # return [f'head:{entities[head]},tail:{entities[tail]},relation:{relation_text}' for head, tail, relation_text in zip(heads, tails, relation_texts)]
    # return [f'head:{head},tail:{tail},relation:{relation_text}' for head, tail, relation_text in zip(heads, tails, relation_texts)]
    return [(entities[head], entities[tail], relation_text) for head, tail, relation_text in zip(heads, tails, relation_texts)]
        
def name_entity_clustering_inout(item):
    doc = sents2doc(item['sents'])
    vertex = item['vertexSet']
    labels = item['labels']
    # name entity clustering
    input_nec = f'{doc}' + ' ' + 'Name entity clusters: '
    output_nec = vertexset2output(vertex)
    # relation extraction
    # history = input_nec + output_nec + '. '
    # for head, tail, relation_text in label2triple(labels, vertex):
    #     input_re = history + 'Relation between [{head}] and [{tail}] is: '
    #     output_re = relation_text
    return input_nec, output_nec
    

def train_step_with_model(m, item):
    input_nec, output_nec = name_entity_clustering_inout(item)
    input_ids = m.toker(input_nec, return_tensors="pt").input_ids.cuda()
    labels = m.toker(output_nec, return_tensors="pt").input_ids.cuda()
    loss = m.t5(input_ids=input_ids, labels=labels).loss
    loss.backward()
    m.opter.step()
    m.opter.zero_grad()
    m.t5.zero_grad()


def train(m, train_ds, logging = True):
    # train
    for index, item in enumerate(train_ds):
        train_step_with_model(m, item)
        if logging and (index + 1) % 500 == 0:
            print(f'{index + 1} / {len(train_ds)}')


def small_script(m, epoch):
    start_time = time.time()
    for i in range(epoch):
        train(m, dataset['train_annotated'])
    print("--- %s seconds ---" % (time.time() - start_time))

def generate(m, dstest, idx):
    input_nec, output_nec = name_entity_clustering_inout(dstest[idx])
    input_ids = m.toker(input_nec, return_tensors="pt").input_ids.cuda()
    outputs = m.t5.generate(input_ids)
    label_text = m.toker.decode(outputs[0], skip_special_tokens=True)
    return label_text




