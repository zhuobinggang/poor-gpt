# 每日新闻处理脚本
from importlib import reload
import re
import random

the_path = 'data/mai2019.utf8.txt'

def read_lines():
  with open(the_path) as fin:
    lines = fin.readlines()
  return lines
  
def get_articles_raw():
  articles = []
  lines = read_lines()
  starts = [idx for idx, line in enumerate(lines) if line.startswith('＼ＡＤ＼')]
  borders = list(zip(starts, starts[1:])) # 丢弃最后一个文本
  for start, end in borders:
    articles.append(lines[start: end])
  return articles

def filter1_start_line_raws(start_line): 
  return start_line.startswith('＼ＡＤ＼０１') or start_line.startswith('＼ＡＤ＼０２') or start_line.startswith('＼ＡＤ＼０３') or start_line.startswith('＼ＡＤ＼１０')
  
# 1. 将所有AD=01(1面), 02(2面), 03(3面), 10(特集)的去掉
def special_type_articles_filtered(articles):
  return [art for art in articles if not filter1_start_line_raws(art[0])]

def except_t2_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([line for line in art if line.startswith('＼Ｔ２＼')])
  return new_articles

def special_tokens_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([remove_special_tokens(line) for line in art])
  return new_articles

def line_without_period_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([line for line in art if line.find('。') >= 0])
  return new_articles

def remove_special_tokens(line):
  # special_tokens = ['\u3000', '\n', '＼Ｔ２＼', '「', '」', '（','）', '○', '＜', '＞', '◆', '〓', '｝', '｛', '■']
  special_tokens = ['\u3000', '\n', '＼Ｔ２＼', '（','）', '○', '＜', '＞', '◆', '〓', '｝', '｛', '■', '●', '◇', '▽']
  special_tokens.append('$') # you can add what you want
  result = line
  for token in special_tokens:
    result = result.replace(token, '')
  return result

def art_with_paragraph_less_then_num_removed(articles, num = 2): 
  return [art for art in articles if not len(art) < num]

def build_structure(articles):
  new_articles = []
  for art in articles:
    sentences = []
    for paragrath in art:
      new_ss = paragrath.split('。')
      new_ss = new_ss[:-1] # 排除最后一个元素，排除像【ワシントン中井正裕】这种东西
      new_ss = [s for s in new_ss if len(s) > 1] # 排除长度过小的句子包括''
      new_ss = [s + '。' for s in new_ss] # 重新补充句号
      if len(new_ss) > 0:
        new_ss[0] = '\u3000' + new_ss[0] # 为了适应以前的数据集
        sentences += new_ss
      else:
        print(f'WARN: Empty paragrath! {paragrath}')
    new_articles.append(sentences)
  return new_articles
      
def line_with_special_token_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([line for line in art if (line.find('】') == -1 and line.find('【') == -1 and line.find('＜') == -1 and line.find('＞') == -1 and line.find('◆') == -1)])
  return new_articles

def paragraph_only_one_sentence_removed(articles):
  results = []
  for art in articles:
    paras = []
    for para in art:
      if len(para.split('。')) <= 2:
        pass
      else:
        paras.append(para)
    results.append(paras)
  return results

def paragraph_with_special_token_removed(articles):
  results = []
  for art in articles:
    paras = [para for para in art if para.find('【') == -1]
    results.append(paras)
  return results

# trim '【ニューヨーク國枝すみれ、ロサンゼルス長野宏美】トランプ米大統領に抗議す'
def special_pattern_removed(arts):
  results = []
  for art in arts:
    paras = [re.sub(r'【.*?】', '', para) for para in art]
    results.append(paras)
  return results

def stand_process_before_build():
  articles = get_articles_raw()
  articles = special_type_articles_filtered(articles) # NO.1
  articles = except_t2_removed(articles)
  articles = line_without_period_removed(articles) # 必要操作： 没有句号的段落需要移除
  articles = special_tokens_removed(articles) # 必要操作： 移除特殊符号，否则模型可能会依赖这些符号
  articles = special_pattern_removed(articles) # 必要操作： 移除特殊模式，否则模型可能会依赖这些符号
  # articles = line_with_special_token_removed(articles) # 非必要操作： 包含特殊token的段落移除。可能会破坏语境连贯性
  # articles = paragraph_only_one_sentence_removed(articles) # 非必要操作： 移除只有一句话的段落，没必要
  articles = art_with_paragraph_less_then_num_removed(articles, num = 2) # 必要操作：有些段落文章只有['この記事は本文を表示できません。']这样一句话
  # articles = paragraph_with_special_token_removed(articles) # 非必要操作：移除包含特殊符号的段落，可能会破坏语境连贯性
  return articles

def standard_process():
  articles = build_structure(stand_process_before_build())
  return articles

# output = 5
def avg_sentence_len(articles):
  lengths = [len(art) for art in articles]
  return int(sum(lengths) / len(lengths))
    
def customize_my_dataset_and_save(structed_articles):
  flated = [item for sublist in structed_articles for item in sublist]
  train = flated[:10000]
  test = flated[10000:15000]
  with open('train.txt', 'a') as the_file:
    for line in train:
      the_file.write(f'{line}\n')
  with open('test.txt', 'a') as the_file:
    for line in test:
      the_file.write(f'{line}\n')

def no_line_breaks(texts):
  return [text.replace('\n', '') for text in texts]

def read_trains(mini = False):
  file_path = 'train.mini.txt' if mini else 'train.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_tests(mini = False):
  file_path = 'test.mini.txt' if mini else 'test.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_valid(mini = False):
  file_path = 'valid.mini.txt' if mini else 'valid.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_trains_big(mini = False):
  file_path = 'train.big.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_tests_big(mini = False):
  file_path = 'test.big.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

# 合并

def get_one_art_per_line(structed_articles):
    one_art_per_line = []
    for art in structed_articles:
        lines = [line.replace('$', '') for line in art]
        art_line = '$'.join(lines)
        one_art_per_line.append(art_line)
    return one_art_per_line

def customize_my_dataset_and_save(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    train = one_art_per_line[2000:4000]
    test = one_art_per_line[4000:4500]
    dev = one_art_per_line[4500:5000]
    manual_exp = one_art_per_line[5000:5500]
    # valid = one_art_per_line[2000:2500]
    with open('data/train.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(train))
    with open('data/test.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(test))
    with open('data/dev.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(dev))
    with open('data/manual_exp.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(manual_exp))

def write_additional_test_datasets(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    tests = []
    tests.append(one_art_per_line[5500:6000])
    tests.append(one_art_per_line[6000:6500])
    tests.append(one_art_per_line[6500:7000])
    tests.append(one_art_per_line[7000:7500])
    tests.append(one_art_per_line[7500:8000])
    tests.append(one_art_per_line[8500:9000])
    tests.append(one_art_per_line[9000:9500])
    tests.append(one_art_per_line[9500:10000])
    tests.append(one_art_per_line[10000:10500])
    tests.append(one_art_per_line[10500:11000])
    for i, test_ds in enumerate(tests):
        with open(f'datasets/test{i}.paragraph.txt', 'w') as the_file:
            the_file.write('\n'.join(test_ds))

def ld_without_opening(ld):
    ld = [case for case in ld if case[0][2] != 0]
    return ld

def read_additional_test_ds():
    tlds = []
    for i in range(10):
        tld = load_customized_loader(file_name = f'test{i}', half = 2, batch = 1, shuffle = False)
        tld = ld_without_opening(tld)
        tlds.append(tld)
    return tlds

def only_label_without_opening(name):
    tld = load_customized_loader(file_name = name, half = 2, batch = 1, shuffle = False)
    tld = ld_without_opening(tld)
    return [case[0][1][case[0][2]] for case in tld]

def read_additional_test_dataset_targets():
    ress = []
    for i in range(10):
        ress.append(only_label_without_opening(f'test{i}'))
    return ress

def read_train_dev_targets():
    return only_label_without_opening('train'), only_label_without_opening('dev')


def customize_my_dataset_and_save_mini(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    train = one_art_per_line[0:300]
    test = one_art_per_line[300:450]
    dev = one_art_per_line[450:600]
    valid = one_art_per_line[600:750]
    with open('data/train.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(train))
    with open('data/test.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(test))
    with open('data/dev.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(dev))
    with open('data/valid.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(valid))

def read_sentences_per_art(path):
    with open(path, 'r') as the_file:
        lines = the_file.readlines()
    arts = [line.split('$') for line in lines]
    return arts

# =============== Analysis Methods ==================

def cal_para_count(arts):
    counts = []
    for sentences in arts:
        counts.append(sum([1 for s in sentences if s.startswith('\u3000')]))
    return counts


# =============== 全新数据集创建逻辑 ================

def is_begining(s):
  return s.startswith('\u3000')

def no_indicator(s):
  return s.replace('\u3000', '')

def read_chapters(file_name = 'train'):
    arts = read_sentences_per_art(f'data/{file_name}.paragraph.txt')
    arts_without_linebreak = []
    for art in arts:
        art = [line.replace(' ', '').replace('\n', '').replace('\r', '') for line in art]
        arts_without_linebreak.append(art)
    return arts_without_linebreak

def create_loader_from_chapters(chapters, window_size = 4):
    loader = []
    assert window_size % 2 == 0
    half_window_size = int(window_size / 2)
    for sentences in chapters:
        length = len(sentences)
        end = len(sentences)
        for center_idx in range(1, length):
            ss = []
            labels = []
            for idx in range(center_idx - half_window_size, center_idx + half_window_size): # idx = 2 时候 range(0, 4)
                if idx < 0 or idx >= length:
                    labels.append(None) # NOTE: 一定要handle None
                    ss.append(None)
                else:
                    s = sentences[idx]
                    labels.append(1 if is_begining(s) else 0)
                    ss.append(no_indicator(s)) # NOTE: 一定要去掉段落开头指示器
            loader.append((ss, labels))
    # NOTE: ASSERT
    count = sum([len(sentences) - 1 for sentences in chapters])
    assert len(loader) == count
    return loader

def read_ld_train(window_size = 4):
    return create_loader_from_chapters(read_chapters('train'), window_size)

def read_ld_test(window_size = 4):
    return create_loader_from_chapters(read_chapters('test'), window_size)

def read_ld_tests(window_size = 4):
    lds = []
    for i in range(10):
        lds.append(create_loader_from_chapters(read_chapters(f'test{i}'), window_size))
    return lds

def read_ld_dev(window_size = 4):
    return create_loader_from_chapters(read_chapters('dev'), window_size)
