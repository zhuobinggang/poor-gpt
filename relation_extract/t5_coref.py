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
    res.toker = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}", truncation_side= 'left')
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

def load_ds(load_train = True):
    loader = Ontonotes()
    if load_train:
        train = list(loader.dataset_document_iterator(dspath['train']))
        test = list(loader.dataset_document_iterator(dspath['test']))
    else:
        train = None
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
    for s in item: 
        words = s.words.copy()
        for mention in s.coref_spans:
            cluster_id, (start, end) = mention
            words[start] = f'[{cluster_id} {words[start]}'
            words[end] = f'{words[end]}]'
        print(s.speakers[0] + ': ' + ' '.join(words) + ' ')
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

def split_and_trim(text, mark):
    lst = [term for term in [s.strip() for s in text.split(mark)] if term != '']
    term = lst[0]
    suffix = lst[1] if len(lst) >1 else ''
    return term, suffix


def contains(big, small):
    for i in reversed(range(len(big)-len(small)+1)): #NOTE: From end to start
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return None

def set_or_create(dic, key, key2, value):
    # print(f'{idx} {prefix} {suffix}')
    if key not in dic:
        dic[key] = {}
    dic[key][key2] = value


# NOTE: 唯一改变的是mark
def output_text_process_step_by_step(context_words, step_output_text, mark, cluster_idx):
    should_increase_cluster_idx = False
    step = step_output_text
    if step == 'SHIFT':
        print(f'OUTPUT_PROCESS(SHIFT)')
    else:
        current, target = split_and_trim(step, '->')
        if target.startswith('['): # Append, 只需要给当前注目的tokens增加标记
            words, suffix = split_and_trim(current, '##')
            words = words.split()
            suffix = suffix.split()
            find_myself = contains(context_words, words + suffix) # 在context_words中找到自己的位置，用于标记。比较tricky因为需要解码模型的文字输出
            if find_myself is None:
                print(f'Wrong! Can not find current words from context! \n {words + suffix}')
            else:
                start, end = find_myself
                end -= len(suffix)
                cluster_idx = int(target.strip('['))
                set_or_create(mark, start, 'prefix', f'[{cluster_idx}')
                set_or_create(mark, end-1, 'suffix', f']')
                # print(f'OUTPUT_PROCESS(APPEND): {words} ## {suffix}, 标记 {context_words[start]} ~ {context_words[end-1]} ')
        else: # LINK, 需要同时给过去和现在的token增加标记
            # print('LINK')
            should_increase_cluster_idx = True
            # print(f'OUTPUT_PROCESS(LINK): and cluster index increased, now = {cluster_idx + 1}')
            # current
            words, suffix = split_and_trim(current, '##')
            words = words.split()
            suffix = suffix.split()
            find_myself = contains(context_words, words + suffix)
            if find_myself is None:
                print(f'Wrong! Can not find current words from context! \n {words + suffix}')
            else:
                start, end = find_myself
                end -= len(suffix)
                set_or_create(mark, start, 'prefix', f'[{cluster_idx + 1}')
                set_or_create(mark, end-1, 'suffix', f']')
                # print(f'(LINK BASE): {words} ## {suffix}, 标记 {context_words[start]} ~ {context_words[end-1]} ')
            # target
            words, suffix = split_and_trim(target, '##')
            words = words.split()
            suffix = suffix.split()
            find_myself = contains(context_words, words + suffix)
            if find_myself is None:
                print(f'Wrong! Can not find target words from context! \n {words + suffix}')
            else:
                start, end = find_myself
                end -= len(suffix)
                set_or_create(mark, start, 'prefix', f'[{cluster_idx + 1}')
                set_or_create(mark, end-1, 'suffix', f']')
                # print(f'(LINK TARGET): {words} ## {suffix}, 标记 {context_words[start]} ~ {context_words[end-1]} ')
                # print(f'LINK_target: {words + suffix} = {context_words[start]} ~ {context_words[end-1]} ')
                # print(mark)
    return should_increase_cluster_idx
    
s_wrong = None

# Prepare for the training set
def process_training_data(document):
    global s_wrong
    cluster_dic = {} # 每次mention创建一个位置，key=cid
    # clusters = [] # 根据模型输出创建和更新的clusters，每次LINK更新下标
    input_outputs = [] # 训练用
    context_words = [] # 就是s.words摊开，包含当前处理的句子单词
    cluster_idx_increase = 0 # 因为根据ouput来处理的时候cluster_idx是顺序递增的，跟cid不同
    mark = {} # (prefix, suffix) # {token_id: {prefix, suffix}}
    for idx, s in enumerate(document):
        # 将说话人添加到mark[idx]
        set_or_create(mark, len(context_words), 'speaker', s.speakers[0])
        context_words += s.words
        # NOTE: Sort as paper: Coreference Resolution through a seq2seq Transition-Based System
        for cid, (start, end) in list(sorted(s.coref_spans.copy(), key = lambda item: item[1][1])):
            current_item_text = ' '.join(s.words[start:end+1]) # OK
            current_item_three_gram_suffix = ' '.join(s.words[end+1:end+4]) # OK
            if cid not in cluster_dic: # Mention
                cluster_dic[cid] = [(cid, current_item_text, current_item_three_gram_suffix)]
                # print(f'MENTION, cluster_idx_increase: {cluster_idx_increase}, OUTPUT: {output_text}')
            else: # Link or Append
                if len(cluster_dic[cid]) == 1: # LINK
                    # TODO: 确保cluster_idx_increase += 1
                    _, previou_item_text, previou_item_three_gram_suffix = cluster_dic[cid][0]
                    input_text = compress_context_words_and_marks(context_words, mark) # 单个句子里面可能会多次更新mark，造成input_text的变更。这说明input_output是对应于command而非句子的
                    output_text = f'{current_item_text} ## {current_item_three_gram_suffix} -> {previou_item_text} ## {previou_item_three_gram_suffix}'
                    input_outputs.append((input_text, output_text))
                    try:
                        should_increase = output_text_process_step_by_step(context_words, output_text, mark, cluster_idx_increase)
                    except ValueError as e:
                        s_wrong = s
                        print(f'{idx}: {output_text}')
                        raise e
                    assert should_increase is True
                    if should_increase:
                        cluster_idx_increase += 1
                    cluster_dic[cid].append((cid, current_item_text, current_item_three_gram_suffix))
                else: # Append
                    input_text = compress_context_words_and_marks(context_words, mark) # 单个句子里面可能会多次更新mark，造成input_text的变更。这说明input_output是对应于command而非句子的
                    output_text = f'{current_item_text} ## {current_item_three_gram_suffix} -> [{cluster_idx_increase}'
                    input_outputs.append((input_text, output_text))
                    try:
                        should_increase = output_text_process_step_by_step(context_words, output_text, mark, cluster_idx_increase)
                    except ValueError as e:
                        s_wrong = s
                        print(f'{idx}: {output_text}')
                        raise e
                    assert should_increase is False
                    if should_increase:
                        cluster_idx_increase += 1
                    cluster_dic[cid].append((cid, current_item_text, current_item_three_gram_suffix))
        # SHIFT
        input_text = compress_context_words_and_marks(context_words, mark) # 单个句子里面可能会多次更新mark，造成input_text的变更。这说明input_output是对应于command而非句子的
        output_text = 'SHIFT'
        input_outputs.append((input_text, output_text))
        # combine output text
    return input_outputs


def compress_context_words_and_marks(context_words, mark):
    text = ''
    for idx, word in enumerate(context_words):
        if idx in mark:
            item = mark[idx]
            prefix = item['prefix'] + ' ' if 'prefix' in item else ''
            suffix = item['suffix'] if 'suffix' in item else ''
            speaker = item['speaker'] + ': ' if 'speaker' in item else ''
            text += f' {speaker}{prefix}{word}{suffix}'
        else:
            text += f' {word}'
    return text

def train(m, ds):
    for doc in ds:
        input_outputs = process_training_data(doc)
        for input_text, label_text in input_outputs:
            input_ids = m.toker(input_text, return_tensors="pt", truncation = True).input_ids.cuda()
            labels = m.toker(label_text, return_tensors="pt").input_ids.cuda()
            loss = m.t5(input_ids=input_ids, labels=labels).loss
            loss.backward()
            m.opter.step()
            m.opter.zero_grad()
            m.t5.zero_grad()

doc_wrong = None

# m = create_model(learning_rate = 2e-5, size = 'small')

def script(m):
    global doc_wrong
    start_time = time.time()
    ds0, ds1 = load_ds()
    for idx, doc in enumerate(ds0):
        if (idx + 1) % 500 == 0: # Log
            print(f'{idx} / {len(ds0)}')
        try: 
            input_outputs = process_training_data(doc)
        except ValueError as e:
            print(idx)
            doc_wrong = doc
            raise e
        for input_text, label_text in input_outputs:
            input_ids = m.toker(input_text, return_tensors="pt", truncation = True).input_ids.cuda()
            labels = m.toker(label_text, return_tensors="pt").input_ids.cuda()
            loss = m.t5(input_ids=input_ids, labels=labels).loss
            loss.backward()
            m.opter.step()
            m.opter.zero_grad()
            m.t5.zero_grad()
    print("--- %s seconds ---" % (time.time() - start_time))
    return m


