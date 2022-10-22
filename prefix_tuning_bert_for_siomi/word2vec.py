import fasttext
ft = fasttext.load_model('/usr01/taku/cc.ja.300.bin')

def get_vec(word):
    return ft.get_word_vector(word)


