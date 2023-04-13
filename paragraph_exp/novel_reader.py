import random
import numpy as np
import re

def read_docs(data_id = 1):
  if data_id == 1:
    filenames = ['sansirou', 'sorekara', 'mon', 'higan', 'gyoujin'] 
  elif data_id == 2:
    filenames = ['kokoro'] 
  elif data_id == 3:
    filenames = ['meian'] 
  paths = [f'data/{name}_new.txt' for name in filenames]
  docs = []
  for path in paths:
    with open(path) as f:
      lines = f.readlines()
      docs.append(lines)
  return docs

def read_lines(data_id = 1):
  docs = read_docs(data_id)
  lines = []
  for doc in docs:
    lines += doc
  return [line.replace(' ', '').replace('\n', '').replace('\r', '') for line in lines] # 去掉空格和换行符

END_CHARACTERS = ['。', '？']

def read_sentences(data_id = 1): 
  lines = read_lines(data_id) # 隐式根据换行划分句子
  sentences = []
  for line in lines:
    s = ''
    for c in line:
      s += c
      if c in END_CHARACTERS:
        sentences.append(s)
        s = ''
    if len(s) > 0:
      if len(s) < 3:
        sentences[-1] += s
      else:
        sentences.append(s)
  return sentences

# 需要根据空行分割成不同的章节
def read_chapters(data_id = 1): 
  lines = read_lines(data_id) # 隐式根据换行划分句子
  chapters = []
  sentences = []
  for line in lines:
      if len(line) == 0: # 空行
          chapters.append(sentences.copy())
          sentences = []
      else:
          s = ''
          for c in line: # 遍历行内所有character
              s += c
              if c in END_CHARACTERS:
                  sentences.append(s)
                  s = ''
          if len(s) > 0: # 处理最后留下来的一点
              if len(s) < 3:
                  sentences[-1] += s
              else:
                  sentences.append(s)
  if len(sentences) > 0:
      print('似乎出了点意外情况，理论上EOF应该是一个空行，所以不应该剩下这个才对')
  return chapters

def read_trains():
  return read_sentences(1)

def read_tests():
  return read_sentences(3)

def read_devs():
  return read_sentences(2)

def read_trains_from_chapters():
  return read_chapters(1)

def read_tests_from_chapters():
  return read_chapters(3)

def read_devs_from_chapters():
  return read_chapters(2)

def no_indicator(s):
  return s.replace('\u3000', '')

def is_begining(s):
  return s.startswith('\u3000')

def cal_label_one(ss):
  return sum([1 for s in ss if is_begining(s)])


class Dataset():
  def __init__(self, ss_len = 8, datas=[]):
    super().__init__()
    self.ss_len = ss_len
    self.datas = datas
    self.init_datas_hook()
    self.init_hook()
    self.start = 0

  def init_hook(self):
    pass

  def init_datas_hook(self):
    pass

  def set_datas(self, datas):
    self.datas = datas

  def is_begining(self, s):
    return s.startswith('\u3000')
        
  def no_indicator(self, ss):
    return [s.replace('\u3000', '') for s in ss]

  # 作为最底层的方法，需要保留所有分割信息
  def get_ss_and_labels(self, start): 
    end = min(start + self.ss_len, len(self.datas))
    start = max(start, 0)
    ss = []
    labels = []
    for i in range(start, end):
      s = self.datas[i]
      labels.append(1 if self.is_begining(s) else 0)
      ss.append(s)
    ss = self.no_indicator(ss)
    return ss, labels

  def __getitem__(self, start):
    return self.get_ss_and_labels(start)

  def __len__(self):
    return len(self.datas)

  def shuffle(self):
    random.shuffle(self.datas)


def train_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len)
  ds.set_datas(read_trains())
  return ds

def test_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len)
  ds.set_datas(read_tests())
  return ds

def dev_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len)
  ds.set_datas(read_devs())
  return ds

# start ======================= Loader Tested, No Touch =======================
class Loader():
  def __init__(self, ds, half, batch):
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2 + 1
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()

  def __iter__(self):
    return self

  def __len__(self):
    return self.end_point() - self.start_point() + 1

  def start_point(self):
    return 0

  def end_point(self):
    return len(self.ds.datas) - 1

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    start = idx - self.half # 可能是负数
    ss, labels = self.ds[start] # 会自动切掉负数的部分
    correct_start = max(start, 0)
    pos = idx - correct_start
    return ss, labels, pos # 只需要中间的label

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  # raise StopIteration()
  def __next__(self):
    start = self.start
    if start > self.end_point():
      self.start = self.start_point()
      raise StopIteration()
    else:
      results = []
      end = min(start + self.batch - 1, self.end_point())
      for i in range(start, end + 1):
        ss, label, pos = self.get_data_by_index(i)
        results.append((ss, label, pos))
      self.start = end + 1
      return results

  def shuffle(self):
    self.ds.shuffle()


class Loader_Symmetry(Loader):
  def __init__(self, ds, half, batch):
    # print(f'init Loader_Symmetry half={half}, batch={batch}')
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    start = idx - self.half # 可能是负数
    ss, labels = self.ds[start] # 会自动切掉负数的部分
    correct_start = max(start, 0)
    pos = idx - correct_start
    return ss, labels, pos # 只需要中间的label


class Loader_SGD():
  def __init__(self, ds, half, batch, shuffle = True):
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2 + 1
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()
    ld = Loader(ds, half, batch=1)
    self.masses = []
    for mass in ld:
      ss, labels, pos = mass[0]
      self.masses.append((ss, labels, pos))
    if shuffle:
      self.shuffle()
      print('Loader_SGD: Shuffled')
    else:
      print('Loader_SGD: No Shuffle')

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.masses)

  def start_point(self):
    return 0

  def end_point(self):
    return len(self.masses) - 1

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    return self.masses[idx]

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  # raise StopIteration()
  def __next__(self):
    start = self.start
    if start > self.end_point():
      self.start = self.start_point()
      raise StopIteration()
    else:
      results = []
      end = min(start + self.batch - 1, self.end_point())
      for i in range(start, end + 1):
        ss, label, pos = self.get_data_by_index(i)
        results.append((ss, label, pos))
      self.start = end + 1
      return results

  def shuffle(self):
    random.shuffle(self.masses)

class Loader_Symmetry_SGD(Loader_SGD):
  def __init__(self, ds, half, batch, shuffle = True):
    # print(f'init Loader_Symmetry_SGD half={half}, batch={batch}')
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()
    ld = Loader_Symmetry(ds, half, batch=1)
    self.masses = []
    for mass in ld:
      ss, labels, pos = mass[0]
      self.masses.append((ss, labels, pos))
    if shuffle:
      random.shuffle(self.masses)
      # print('Loader_Symmetry_SGD: Shuffled')
    else:
      # print('Loader_Symmetry_SGD: No Shuffle')
      pass
  
# end ======================= Loader Tested, No Touch =======================
# 合并

# NOTE: 因为小说数据集只有头跟尾两个例外，直接无视特殊情况
def create_loader(sentences, window_size = 4):
    loader = []
    length_max = len(sentences)
    start = 0
    end = length_max - window_size + 1 # (+1代表包含)
    half_window = int(window_size / 2)
    for idx in range(start, end):
        ss = []
        labels = []
        for left_idx in range(idx, idx + window_size): # (0, 4)
            s = sentences[left_idx]
            labels.append(1 if is_begining(s) else 0)
            ss.append(no_indicator(s)) # NOTE: 一定要去掉段落开头指示器
        loader.append((ss, labels))
    assert len(loader) == (length_max - window_size + 1)
    return loader

def read_ld_train():
    return create_loader(read_trains(), 4)

def read_ld_test():
    return create_loader(read_tests(), 4)


def read_ld_dev():
    return create_loader(read_devs(), 4)

# =============================================================================

# NOTE: 要考虑章节接续点
def read_ld_train_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_trains_from_chapters(), window_size)

def read_ld_test_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_tests_from_chapters(), window_size)

# NOTE: 用来计算t值 
# 读取188个章节，每个章节一个loader
def read_lds_test_from_chapters(window_size = 4):
    chapters = read_tests_from_chapters()
    lds = []
    for chapter in chapters:
        lds.append(create_loader_from_chapters([chapter], window_size))
    return lds

def read_ld_dev_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_devs_from_chapters(), window_size)

# NOTE: 要考虑章节接续点
# NOTE: TESTED, window_size = 6的情况也已经测试 
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


def test_create_loader_from_chapters():
    ld = read_ld_dev_from_chapters(window_size = 6)
    print(ld[37])
    print(ld[38])
    print(ld[-1])
    
def cal_ones(ds_from_chapter):
    ones = 0
    for ss, labels in ds_from_chapter:
        ones += labels[2]
    return ones
