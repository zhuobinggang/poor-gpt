from lstm_fasttext import *

@torch.no_grad()
def att(model, ss, ls):
    rnn = model.rnn
    mlp = model.mlp
    tagger = model.tagger
    ft = model.ft
    left = combine_ss(ss[:2])
    right = combine_ss(ss[2:])
    left_words = tagger(left)
    right_words = tagger(right)
    left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in left_words]) # (?, 300)
    right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in right_words]) # (?, 300)
    left_vecs, (_, _) = rnn(left_vecs.cuda()) # (?, 600)
    right_vecs, (_, _) = rnn(right_vecs.cuda()) # (?, 600)
    assert len(left_vecs.shape) == 2 
    assert left_vecs.shape[1] == 600
    # Attention
    left_att = model.attention(left_vecs).view(-1).tolist()
    right_att = model.attention(right_vecs).view(-1).tolist()
    # return list(zip(left_words, left_att)), list(zip(right_words, right_att))
    return [str(word) for word in left_words + [' 口口口 '] + right_words], [round(val, 4) for val in left_att + [0] + right_att]

model = load_checkpoint('checkpoint/att_e5_f0.568.checkpoint')
ld_test = loader.news.test()
ss, ls = ld_test[101]


################## BERT ###################
## 放弃了，感觉可解释性有点差，层数太多了

from sector import Sector

PATH = 'checkpoint/bert_e4_f0.665.checkpoint'
model = Sector()
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

ss, ls = ld_test[101]
combined_ids, _ = encode_standard(ss, model.toker)
attention = model.bert(combined_ids.unsqueeze(0).cuda(), output_attentions = True).attentions 
# tuple of (layers = 12, batch = 1, heads = 12, seq_len = 138, seq_len = 138)
tokens = model.toker.convert_ids_to_tokens(combined_ids)


