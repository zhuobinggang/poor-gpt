import torch
from torch_geometric.data import Data
from data_loader import loader
import fasttext
from fugashi import Tagger
from common import combine_ss

def edge_create(length):
    edge_index_raw = []
    for i in range(0, length - 1):
        edge_index_raw.append([i, i+1])
        edge_index_raw.append([i+1, i])
        if i+2 < length:
            edge_index_raw.append([i, i+2])
            edge_index_raw.append([i+2, i])
    return torch.tensor(edge_index_raw, dtype=torch.long)

def create_data(x):
    edge_index = edge_create(len(x)).t()
    data = Data(x=x, edge_index=edge_index)
    return data

def script():
    ft = fasttext.load_model('cc.ja.300.bin')
    tagger = Tagger('-Owakati')
    ld_train = loader.news.train()
    ss, ls = ld_train[0]
    left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(combine_ss(ss[:2]))]) # (?, 300)
    right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(combine_ss(ss[2:]))]) # (?, 300)
    # Left graph
    left_data = create_data(left_vecs)
    left_data.validate(raise_on_error=True)
    # Right graph
    right_data = create_data(right_vecs)
    right_data.validate(raise_on_error=True)



