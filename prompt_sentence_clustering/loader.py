import re
import csv
import unicodedata

def different_character_scan(x,y):
    for idx, (c1, c2) in enumerate(zip(x,y)):
        if c1 != c2:
            print(f'{c1} != {c2}, IDX = {idx}')

def find_startswith_in_ss(ss, start):
    for idx, s in enumerate(ss):
        if s.startswith(start):
            return idx
    return -1

def text_get_rid_of_weird_things(text):
    weird_chars = [' ', '　', '、', '，', ',', '。', '.', '。', '−', '-', '’', "'"]
    text = unicodedata.normalize('NFKC',text)
    for weird_char in weird_chars:
        text = text.replace(weird_char, '')
    return text

# NOTE: Only between sentence from doc, and sentence from table
def text_equal(x, y):
    x = text_get_rid_of_weird_things(re.sub('（.*?）', '', x))
    y = text_get_rid_of_weird_things(re.sub('（.*?）', '', y))
    if abs(len(x) - len(y)) < 2:
        if x in y or y in x:
            return True
    return False

def find_in_nest_lst_soft(lst, item):
    for current_idx, current_items in enumerate(lst):
        for current_item in current_items:
            if text_equal(current_item, item):
                return current_idx
    return -1

def find_in_lst_soft(lst, item):
    if isinstance(lst[0], list):
        return find_in_nest_lst_soft(lst, item)
    else:
        for current_idx, current_item in enumerate(lst):
            if text_equal(current_item, item):
                return current_idx
        return -1

def load_docs_old(filename = './data/goutou.txt'):
    f = open(filename)
    lines = f.readlines()
    f.close()
    docs = []
    for line in lines:
        if line.startswith('<doc'):
            current_doc = {}
            title = re.findall('.*? title=\"(.*?)\".*', line)[0]
            current_doc['title'] = title
            current_doc['ss'] = [title]
            docs.append(current_doc.copy()) # old one
        else:
            if line in ['\n', ''] or line.startswith('</doc>'):
                pass
            else:
                current_doc['ss'].append(line.strip()) # NOTE: Strip or not are the same for BERT
    return docs

# Compatible with load_docs_old
def load_docs(filename = 'goutou'):
    filename = f'./data/{filename}.txt'
    f = open(filename)
    lines = f.readlines()
    f.close()
    docs = []
    for line in lines:
        if line.startswith('<doc'):
            current_doc = {}
            title = re.findall('.*? title=\"(.*?)\".*', line)[0]
            current_doc['title'] = title
            current_doc['ss'] = [title]
        elif line.startswith('</doc>'):
            docs.append(current_doc.copy()) # old one
        else:
            if len(line.strip()) < 1:
                pass
            else:
                # TODO: 以。为界区分句子
                for sentence in re.findall('(.*?。|.*$)', line):
                    sentence = sentence.strip()
                    if len(sentence) > 0:
                        current_doc['ss'].append(sentence) # NOTE: Strip or not are the same for BERT
    return docs

def read_label(filename = 'goutou'):
    # filename = '/usr01/taku/datasets/table_cluster/table/GOUTOU_SEIKAI_TABLE.csv'
    # filename = '/usr01/taku/datasets/table_cluster/table/CAMERA_SEIKAI_TABLE.csv'
    filename = f'./data/{filename}.csv'
    f = open(filename)
    rd = csv.reader(f, delimiter=',', quotechar='|')
    col_titles = next(rd)
    # cols = cols[0].split(',')
    rows = []
    for row in rd:
        cols = []
        for col in row:
            col_ss = []
            for s in col.split('|'):
                if s != '':
                    col_ss.append(s)
            cols.append(col_ss)
        rows.append(cols)
    f.close()
    return col_titles, rows

def check_ground_truth_fine(ground_truth, ss, col_names = None):
    for col_idx, col in enumerate(ground_truth):
        for sent_idx, sentence in enumerate(col):
            if sentence != '' and find_in_lst_soft(ss, sentence) == -1:
                if col_names is None:
                    raise ValueError(f'ERR: ({col_idx},{sent_idx}) sentence in cluster but not in the article! sentence: {sentence}')
                else:
                    raise ValueError(f'ERR: Colname: {col_names[col_idx]}, ({col_idx},{sent_idx}) sentence in cluster but not in the article! sentence: {sentence}')
    return True


def csv_check_by_name(name):
    cols, ground_truths = read_label(name)
    csv_check(cols, ground_truths)

def csv_check(cols, rows):
    length = len(cols)
    for idx, row in enumerate(rows):
        if len(row) != length:
            raise ValueError(f'Table wrong! IDX = {idx}, LEN = {len(row)}, ROW = {row}')

def get_docs_and_rows(name):
    docs = load_docs(name)
    cols, rows = read_label(name)
    return docs, cols, rows

def recheck(name):
    docs = load_docs(name)
    cols, ground_truths = read_label(name)
    csv_check(cols, ground_truths)
    assert len(ground_truths) == len(docs)
    # TODO: Check paring is good!
    for idx, (ground_truth, doc) in enumerate(zip(ground_truths, docs)):
        if len(ground_truth) != len(cols):
            raise ValueError(f'WRONG NUMBER of cols! {idx}')
        try:
            check_ground_truth_fine(ground_truth, doc['ss'], cols)
        except ValueError as e:
            print(f'IDX = {idx}')
            raise e

def create_training_data(name = 'goutou'):
    res = []
    docs = load_docs(name)
    cols, ground_truths = read_label(name)
    recheck(name)
    assert len(docs) == len(ground_truths) # Need to pair
    for current_idx in range(1, len(docs)):
        pre_idx = current_idx - 1
        pre_doc = docs[pre_idx]
        current_doc = docs[current_idx]
        # Prepare context
        context = ''
        for idx, s in enumerate(pre_doc['ss']):
            context += f' [{idx}] {s} '
        # Prepare LINK
        ground_truth_me = ground_truths[current_idx] 
        ground_truth_pre = ground_truths[pre_idx]
        for current_sentence in current_doc['ss']:
            prompt = f'{context} | 次の文は[MASK]番の文と似ている: {current_sentence}'
            cluster_idx = find_in_lst_soft(ground_truth_me, current_sentence)
            if cluster_idx >= len(ground_truth_pre):
                print(f'ERR! DOC_IDX: {current_idx}, CLUSTER_IDX: {cluster_idx}, S: {current_sentence}, G: {ground_truth_pre}')
                raise ValueError
            if cluster_idx != -1 and len(ground_truth_pre[cluster_idx]) > 0: # 存在cluster
                # Find index of the sentence in the previous article
                for possible_sentence in ground_truth_pre[cluster_idx]:
                    index_in_prev_doc = find_in_lst_soft(pre_doc['ss'], possible_sentence)
                    if index_in_prev_doc == -1:
                        print(f'ERR! DOC_IDX: {current_idx}, CLUSTER_IDX: {cluster_idx}, S: {current_sentence}')
                        raise ValueError
                    res.append((prompt, index_in_prev_doc))
            else:
                res.append((prompt, -1))
    return res


def step(pre_article, mark, current_sentence):
    pass

def cal_f(outputs, ground_truths):
    pass

