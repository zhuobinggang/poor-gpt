from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import math
import numpy as np
import time
import types
from conll2012_ontonotesv5 import Ontonotes
import itertools


def create_model(learning_rate = 2e-5, size = 'small', max_token_size = 512, t5 = None):
    res = types.SimpleNamespace()
    # NOTE
    # res.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    # res.toker = T5Tokenizer.from_pretrained("google/flan-t5-small")
    if t5 is None:
        res.t5 = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{size}")
    else:
        res.t5 = t5
    res.toker = T5Tokenizer.from_pretrained(f"google/flan-t5-{size}", truncation_side= 'left', model_max_length = max_token_size)
    # res.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    # res.toker = T5Tokenizer.from_pretrained("google/flan-t5-base")
    res.opter = torch.optim.AdamW(res.t5.parameters(), learning_rate)
    res.t5.cuda()
    res.t5.train()
    return res

def save_model(m, name):
    torch.save(m.t5, f'./check_points/{name}.tch')

def load_model(name, size, max_token_size, learning_rate = 2e-5):
    t5 = torch.load(f'./check_points/{name}.tch')
    res = create_model(learning_rate, size, max_token_size, t5)
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
    lst = [s.strip() for s in text.split(mark)]
    lst = [term for term in lst if term != '']
    # raise ValueError(f'split_and_trim error, TEXT: {text}, MARK: {mark}')
    term = '' if len(lst) < 1 else lst[0]
    suffix = lst[1] if len(lst) >1 else ''
    return term, suffix


def contains(big, small):
    for i in reversed(range(len(big)-len(small)+1)): #NOTE: From end to start
        temp_big = ''.join(big[i:])
        temp_big = temp_big.replace(' ', '') # 解决空格赋予问题
        temp_small = ''.join(small).replace(' ', '') # 解决空格赋予问题
        try: 
            _ = temp_big.index(temp_small)
            # Restruct end
            end = i
            temp_text = ''
            for j in range(i, len(big)):
                temp_text += big[j]
                if len(temp_text) >= len(temp_small):
                    return i, j
            print(f'SHOULD NOT COME HERE, {temp_small}, {temp_big}')
            return i, i + len(small)
        except ValueError as e:
            continue
        # for j in range(len(small)):
        #     if big[i+j] != small[j]:
        #         break
        # else:
        #     return i, i+len(small)
    return None

def set_or_create(dic, key, key2, value):
    # print(f'{idx} {prefix} {suffix}')
    if key not in dic:
        dic[key] = {}
    dic[key][key2] = value


# NOTE: 唯一改变的是mark, cluster_dic_of_model
# return_state = (0: SHIFT, 1: LINK, 2: APPEND)
def output_text_process_step_by_step(context_words, step_output_text, mark, cluster_idx, cluster_dic_of_model = None, return_state = False):
    step = step_output_text
    if step == 'SHIFT':
        # print(f'OUTPUT_PROCESS(SHIFT)')
        return (False, 0) if return_state else False
    else:
        # TODO: 分解[,]不让他和前面的单词连在一起
        step = ' , '.join(step.split(','))
        current, target = split_and_trim(step, '->')
        if target.startswith('['): # Append, 只需要给当前注目的tokens增加标记
            words, suffix = split_and_trim(current, '##')
            words = words.split()
            suffix = suffix.split()
            find_myself = contains(context_words, words + suffix) # 在context_words中找到自己的位置，用于标记。比较tricky因为需要解码模型的文字输出
            if find_myself is None:
                print(f'Wrong WHEN APPEND! Can not find current words from context!')
                print(step_output_text)
                print(' '.join(context_words[-30:]))
                return (False, 0) if return_state else False
            else:
                start, end = find_myself
                end -= len(suffix)
                try: 
                    cluster_idx = int(target.strip('['))
                except ValueError as e:
                    print(f'WRONG CLUSTER INDEX!!')
                    print(step_output_text)
                    print(' '.join(context_words[-30:]))
                    return (False, 0) if return_state else False

                set_or_create(mark, start, 'prefix', f'[{cluster_idx}')
                set_or_create(mark, end-1, 'suffix', f']')
                if cluster_dic_of_model is not None:
                    if cluster_idx not in cluster_dic_of_model:
                        print(f'WRONG WHEN APPEND!! cluster_idx {cluster_idx} not in cluster_dic_of_model.')
                        print(step_output_text)
                        print(' '.join(context_words[-30:]))
                        return (False, 0) if return_state else False
                    else:
                        current_item_text = ' '.join(words)
                        current_item_three_gram_suffix = ' '.join(suffix)
                        item_current = (cluster_idx, current_item_text, current_item_three_gram_suffix)
                        cluster_dic_of_model[cluster_idx].append(item_current)
                        return (False, 2) if return_state else False # 2 = APPEND
        else: # LINK, 需要同时给过去和现在的token增加标记
            # current
            words, suffix = split_and_trim(current, '##')
            words = words.split()
            suffix = suffix.split()
            find_myself = contains(context_words, words + suffix)
            if find_myself is None:
                print(f'Wrong WHEN LINK! Can not find current words from context!')
                print(step_output_text)
                print(' '.join(context_words[-30:]))
                return (False, 0) if return_state else False
            else:
                start_cur, end_cur = find_myself
                end_cur -= len(suffix)
                current_item_text = ' '.join(words)
                current_item_three_gram_suffix = ' '.join(suffix)
            # target
            words, suffix = split_and_trim(target, '##')
            words = words.split()
            suffix = suffix.split()
            find_myself = contains(context_words, words + suffix)
            if find_myself is None:
                print(f'Wrong WHEN LINK! Can not find target words from context!')
                print(step_output_text)
                print(' '.join(context_words[-30:]))
                return (False, 0) if return_state else False
            else:
                start_tar, end_tar = find_myself
                end_tar -= len(suffix)
                target_item_text = ' '.join(words)
                target_item_three_gram_suffix = ' '.join(suffix)
            set_or_create(mark, start_cur, 'prefix', f'[{cluster_idx + 1}')
            set_or_create(mark, end_cur-1, 'suffix', f']')
            set_or_create(mark, start_tar, 'prefix', f'[{cluster_idx + 1}')
            set_or_create(mark, end_tar-1, 'suffix', f']')
            # Modify dic_cluster
            if cluster_dic_of_model is not None:
                item_current = (cluster_idx + 1, current_item_text, current_item_three_gram_suffix)
                item_target = (cluster_idx + 1, target_item_text, target_item_three_gram_suffix)
                cluster_dic_of_model[cluster_idx + 1] = [item_current, item_target]
            return (True, 1) if return_state else True # (True = should add cluster, 1 = LINK)
    raise ValueError('WRONG: SHOULD NOT COME HERE!!!')
    
s_wrong = None

# Prepare for the training set
def process_training_data(document, need_ground_truth_dic = False):
    global s_wrong
    cluster_dic = {} # 每次mention创建一个位置，key=cid
    # clusters = [] # 根据模型输出创建和更新的clusters，每次LINK更新下标
    input_outputs = [] # 训练用
    context_words = [] # 就是s.words摊开，包含当前处理的句子单词
    cluster_idx_increase = 0 # 因为根据ouput来处理的时候cluster_idx是顺序递增的，跟cid不同
    mark = {} # (prefix, suffix) # {token_id: {prefix, suffix}}
    cluster_dic_ground_truth = {}
    for idx, s in enumerate(document):
        # 将说话人添加到mark[idx]
        if s.speakers[0] is not None:
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
                        should_increase = output_text_process_step_by_step(context_words, output_text, mark, cluster_idx_increase, cluster_dic_ground_truth)
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
                        should_increase = output_text_process_step_by_step(context_words, output_text, mark, cluster_idx_increase, cluster_dic_ground_truth)
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
    if need_ground_truth_dic:
        return input_outputs, cluster_dic_ground_truth
    else: 
        return input_outputs


def compress_context_words_and_marks(context_words, mark):
    text = ''
    for idx, word in enumerate(context_words):
        if idx in mark:
            item = mark[idx]
            prefix = (item['prefix'] + ' ') if 'prefix' in item else ''
            suffix = item['suffix'] if 'suffix' in item else ''
            try:
                speaker = (item['speaker'] + ': ') if 'speaker' in item else ''
            except TypeError as e:
                print(idx)
                print(context_words)
                print(mark)
                raise e
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

def train_epochs(m, epoch = 1):
    global doc_wrong
    start_time = time.time()
    ds0, ds1 = load_ds()
    for e in range(epoch):
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
        save_model(m, name = f'e{e + 1}')
    print("--- %s seconds ---" % (time.time() - start_time))
    return m

@torch.no_grad()
def test_one_step(m, doc):
    outputs = []
    input_outputs = process_training_data(doc)
    for input_text, label_text in input_outputs:
        input_ids = m.toker(input_text, return_tensors="pt", truncation = True).input_ids.cuda()
        text = m.toker.decode(m.t5.generate(input_ids)[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs

def print_in_out(m, doc):
    outputs = test_one_step(m, doc)
    input_outputs = process_training_data(doc)
    ground_truths = [out_text for in_text, out_text in input_outputs]
    for g,o in zip(ground_truths, outputs):
        if g == o and g != 'SHIFT':
        # if g != o and o != 'SHIFT':
            print(g)
            print(o)
            print('')


###################### For inference ########################

def inference(m, doc):
    cluster_idx_increase = 0 # 因为根据ouput来处理的时候cluster_idx是顺序递增的，跟cid不同
    context_words = [] # 就是s.words摊开，包含当前处理的句子单词
    mark = {} # (prefix, suffix) # {token_id: {prefix, suffix}}
    cluster_dic_of_model = {} # 模型专用的dic
    commands = []
    for idx, s in enumerate(doc):
        # 将说话人添加到mark[idx]
        if s.speakers[0] is not None:
            set_or_create(mark, len(context_words), 'speaker', s.speakers[0])
        context_words += s.words
        # TODO: 因为一句话可以产生多个命令，需要反复确认step_output_text是否为SHIFT
        loop = True
        while loop:
            # Inference by model
            input_text = compress_context_words_and_marks(context_words, mark) # 单个句子里面可能会多次更新mark，造成input_text的变更。这说明input_output是对应于command而非句子的
            input_ids = m.toker(input_text, return_tensors="pt", truncation = True).input_ids.cuda()
            step_output_text = m.toker.decode(m.t5.generate(input_ids)[0], skip_special_tokens=True)
            should_increase_cluster_idx, state= output_text_process_step_by_step(context_words, step_output_text, mark, cluster_idx_increase, cluster_dic_of_model, return_state = True)
            commands.append(step_output_text)
            loop = False if state == 0 else True
    return cluster_dic_of_model, commands

def cluster_dic_to_pairs(dic):
    res = []
    for key in dic:
        lst = [f'{stem} {suffix}' for cid,stem,suffix in dic[key]]
        res += list(itertools.combinations(lst,2))
    return res

def pairs_in(pair, pairs):
    my_term, my_suffix = pair
    my_text = (my_term + my_suffix).replace(' ', '')
    for their_pair in pairs:
        their_term, their_suffix = their_pair
        their_text = (their_term + their_suffix).replace(' ', '')
        if (their_text in my_text) or (my_text in their_text):
            return True
    return False

def cal_paired_f(dic_true, dic_pred):
    c_true = cluster_dic_to_pairs(dic_true)
    c_pred = cluster_dic_to_pairs(dic_pred)
    TP = sum([1 for pair in c_true if pairs_in(pair, c_pred)])
    FN = len(c_true) - TP
    FP = len(c_pred) - TP
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f

def script():
    y_pred = []
    y_true = []
    _, ds_test = load_ds(load_train = False)
    m = load_model('e9', 'base', 128 * 3)
    for doc in ds_test:
        _, ground_truth_dic = process_training_data(doc, need_ground_truth_dic = True)
        with torch.no_grad():
            cluster_dic_of_model, _ = inference(m, doc)
        y_pred.append(cluster_dic_of_model)
        y_true.append(ground_truth_dic)
    ress = []
    for idx, (dic_true, dic_pred) in enumerate(zip(y_true, y_pred)):
        try:
            ress.append(cal_paired_f(dic_true, dic_pred))
        except ZeroDivisionError as e:
            print('ZeroDivisionError: WRONG at idx = {idx}')
            c_true = cluster_dic_to_pairs(dic_true)
            c_pred = cluster_dic_to_pairs(dic_pred)
            print(c_true)
            print(c_pred)
            ress.append((0,0,0))





