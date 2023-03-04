from common import *
from loader import create_training_data, load_docs, read_label, find_in_lst_soft
import itertools

def cal_loss(m, item):
    model = m.bert
    tokenizer = m.toker
    prompt, y = item
    input_ids = tokenizer(prompt, return_tensors="pt", truncation = True).input_ids
    label_text = prompt.replace('[MASK]', str(y) if y != -1 else '？')
    labels = tokenizer(label_text, return_tensors="pt", truncation = True)["input_ids"]
    labels = torch.where(input_ids == tokenizer.mask_token_id, labels, -100)
    outputs = model(input_ids = input_ids.cuda(), labels=labels.cuda())
    return outputs.loss

def dry_run(m, item):
    prompt, y = item
    word = get_predicted_word(m, prompt)
    if word in ['？', '?']:
        return -1, y
    else:
        try: 
            y_pred = int(word)
            return y_pred, y
        except ValueError as e:
            print(f'ERROR!: model should not output {word}!')
            return -1, y

def get_predicted_word(m, prompt):
    tokenizer = m.toker
    model = m.bert
    input_ids = tokenizer(prompt, return_tensors="pt", truncation = True).input_ids
    with torch.no_grad():
        logits = model(input_ids = input_ids.cuda()).logits
    mask_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    return word


######################## 2023.3.1 #########################

def inference(m, docs):
    assert len(docs) == 20
    matrix = np.full((20, 20), -1, dtype=int) # (docs, sentences), assume the number of sentences will not exceed 20
    clusters = []
    cluster_idx_increase = -1
    for current_doc_idx in range(1, len(docs)):
        pre_doc_idx = current_doc_idx - 1
        pre_doc = docs[pre_doc_idx]
        assert len(pre_doc['ss']) < 21
        current_doc = docs[current_doc_idx]
        context = ''
        for idx, s in enumerate(pre_doc['ss']):
            context += f' [{idx}] {s} '
        for current_doc_sentence_idx, current_sentence in enumerate(current_doc['ss']):
            prompt = f'{context} | 次の文は[MASK]番の文と似ている: {current_sentence}'
            pre_doc_sentence_idx = get_predicted_word(m,prompt)
            if pre_doc_sentence_idx in ['?', '？']:
                pre_doc_sentence_idx = -1
            else:
                try:
                    pre_doc_sentence_idx = int(pre_doc_sentence_idx)
                except ValueError as e:
                    print(e)
                    print(f'Wrong output: {pre_doc_sentence_idx}\n Prompt: {prompt}')
                    pre_doc_sentence_idx = -1
            if pre_doc_sentence_idx != -1: # LINK or APPEND
                if pre_doc_sentence_idx >= len(pre_doc['ss']):
                    print(context)
                    print(pre_doc_sentence_idx)
                    raise ValueError(f'{pre_doc_sentence_idx} will out of range!')
                if matrix[pre_doc_idx, pre_doc_sentence_idx] == -1: # LINK
                    cluster_idx_increase += 1
                    matrix[pre_doc_idx, pre_doc_sentence_idx] = cluster_idx_increase # UPDATE MATRIX
                    matrix[current_doc_idx, current_doc_sentence_idx] = cluster_idx_increase # UPDATE MATRIX
                    assert len(clusters) == cluster_idx_increase
                    target_item = (pre_doc_idx, pre_doc_sentence_idx, pre_doc['ss'][pre_doc_sentence_idx])
                    current_item = (current_doc_idx, current_doc_sentence_idx, current_doc['ss'][current_doc_sentence_idx])
                    clusters.append([target_item, current_item]) # # UPDATE CLUSTER LIST
                else: # APPEND
                    cid_temp = matrix[pre_doc_idx, pre_doc_sentence_idx]
                    matrix[current_doc_idx, current_doc_sentence_idx] = cid_temp # UPDATE MATRIX
                    current_item = (current_doc_idx, current_doc_sentence_idx, current_doc['ss'][current_doc_sentence_idx])
                    clusters[cid_temp].append(current_item) # # UPDATE CLUSTER LIST
            else: # SHIFT
                pass
    return clusters, matrix


def ground_true(name):
    docs = load_docs(name)
    col_titles, rows = read_label(name)
    what = {}
    for doc_idx, (doc, row) in enumerate(zip(docs, rows)):
        ss = doc['ss']
        for cid, col in enumerate(row):
            for s in col:
                sentence_idx = find_in_lst_soft(ss, s)
                if sentence_idx != -1:
                    if cid in what:
                        what[cid].append((doc_idx, sentence_idx, s))
                    else:
                        what[cid] = [(doc_idx, sentence_idx, s)]
                else:
                    print('WRONG!')
    what = [what[key] for key in what]
    return what



######################### Calculation ###########################

def cluster_to_pairs(cluster):
    res = []
    for nest_lst in cluster:
        items = [(doc_idx, s_idx) for doc_idx, s_idx, sentence in nest_lst]
        res += list(itertools.combinations(items,2))
    return res

def cal_paired_f(dic_true, dic_pred):
    c_true = cluster_to_pairs(dic_true)
    c_pred = cluster_to_pairs(dic_pred)
    TP = sum([1 for pair in c_true if pair in c_pred])
    FN = len(c_true) - TP
    FP = len(c_pred) - TP
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f

def intersection(list1, list2):
    return list(set(list1).intersection(list2))

def cluster_item_in(lst, item):
    for doc_idx, s_idx, _ in lst:
        if doc_idx == item[0] and s_idx == item[1]:
            return True
    return False

def cal_f(cluster_pred, cluster_true):
    attacks = []
    for focal_cluster in cluster_true:
        temp_focal_cluster = [(doc_idx, s_idx) for doc_idx, s_idx, sentence in focal_cluster]
        attack = []
        for compare_cluster in cluster_pred:
            temp_compare_cluster = [(doc_idx, s_idx) for doc_idx, s_idx, sentence in compare_cluster]
            attack.append(len(intersection(temp_compare_cluster, temp_focal_cluster)))
        attacks.append(attack)
    paired_cluster_pred = [cluster_pred[idx] for idx in  np.argmax(attacks, 1)]
    assert len(paired_cluster_pred) == len(cluster_true)
    ress = []
    for ours, theirs in zip(paired_cluster_pred, cluster_true):
        TP = sum([1 for item in theirs if cluster_item_in(ours, item)])
        FN = len(theirs) - TP
        FP = len(ours) - TP
        prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
        rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
        f = (2 * (prec * rec) / (prec + rec)) if (prec + rec) != 0 else 0
        ress.append((prec, rec, f))
    return ress


######################### Calculation ###########################






