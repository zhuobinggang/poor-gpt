from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
import numpy as np
import time
import types
from conll2012_ontonotesv5 import Ontonotes


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

dspath = {
        'train': '/usr01/taku/datasets/conll_eng_v4/train.english.v4_gold_conll',
        'test': '/usr01/taku/datasets/conll_eng_v4/test.english.v4_gold_conll',
        'dev': '/usr01/taku/datasets/conll_eng_v4/dev.english.v4_gold_conll'
}

def load_ds():
    loader = Ontonotes()
    train = list(loader.dataset_document_iterator(dspath['train']))
    test = list(loader.dataset_document_iterator(dspath['test']))
    return train, test

def transform_document(item):
    text = ''
    doc = item
    for sentence in doc:
        text += (' '.join(sentence.words) + ' ')
    return text

def show_document_with_coref_clustering(item):
    text = ''
    for s in doc: 
        words = s.words.copy()
        for mention in s.coref_spans:
            cluster_id, (start, end) = mention
            words[start] = f'[{cluster_id} {words[start]}'
            words[end] = f'{words[end]}]'
        text += (s.speakers[0] + ': ' + ' '.join(words) + ' ')
    return text


# 按照时间顺序排列
# clusters: [cluster]
# cluster: [(cid, text, three_gram_suffix)]

def find_in_clusters(cid, clusters):
    res_idx = None
    for idx, cluster in enumerate(clusters):
        cid_it, text, suffix = cluster[0]
        if cid == cid_it:
            res_idx = idx
            break
    return res_idx

def sentence_process(s, clusters):
    input_text = ''
    output_texts = []
    words = s.words
    coref_clusters = s.coref_spans.copy()
    # NOTE: Sort as paper: Coreference Resolution through a seq2seq Transition-Based System
    coref_clusters_sorted = list(sorted(coref_clusters, key = lambda item: item[1][1]))
    for cid, (start, end) in coref_clusters_sorted:
        cluster_idx = find_in_clusters(cid, clusters) # 我们自己的id, 按照顺序递增
        ###
        current_item_text = ' '.join(words[start:end+1])
        current_item_three_gram_suffix = ' '.join(words[end+1:end+4])
        if cluster_idx is not None:
            cluster = clusters[cluster_idx]
            if len(cluster) == 1 : # LINK
                _, previou_item_text, previou_item_three_gram_suffix = cluster[0]
                output_text = f'{current_item_text} ## {current_item_three_gram_suffix} -> {previou_item_text} ## {previou_item_three_gram_suffix}'
                output_texts.append(output_text)
                # 添加到cluster
                cluster.append((cid, current_item_text, current_item_three_gram_suffix))
            else: # APPEND
                output_text = f'{current_item_text} ## {current_item_three_gram_suffix} -> [{cluster_idx}'
                output_texts.append(output_text)
                cluster.append((cid, current_item_text, current_item_three_gram_suffix))
        else: # 之前并不存在相同cluster，创建cluster
            clusters.append([(cid, current_item_text, current_item_three_gram_suffix)])
    # SHIFT
    output_texts.append('SHIFT')
    # combine output text
    output_text = ' ; '.join(output_texts)
    return output_text
    
# Prepare for the training set
def process_training_data(document):
    clusters = []
    history_context = ''
    input_outputs = []
    for s in document:
        output_text = sentence_process(s, clusters)
        input_outputs.append((None, output_text))
    return clusters, input_outputs







