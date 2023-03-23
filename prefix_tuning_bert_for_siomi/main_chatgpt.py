from chatGPT import send, only_contents
from reader import read_regular_ds_zip
import numpy as np
import pickle

def transform_ld(ld = None):
    if ld is None:
        ld = read_regular_ds_zip()
    ld_trans = []
    for bad, good in ld:
        label = np.random.randint(1,3)
        if label == 2:
            text = f'Japanese word "{bad}" and "{good}", which one is more commonly used? Answer in 1 or 2, you have to decide.'
        elif label == 1:
            text = f'Japanese word "{good}" and "{bad}", which one is more commonly used? Answer in 1 or 2, you have to decide.'
        else:
            raise ValueError('?????')
        ld_trans.append((text, label))
    return ld_trans

def script():
    bad_good = read_regular_ds_zip()
    ld = transform_ld()
    responses = []
    labels = []
    for request, label in ld:
        responses.append(send(request))
        labels.append(label)
    txt = open('log/chatgpt_responses_230317_N2.txt','wb')
    pickle.dump(responses,txt)
    txt.close()
    contents = only_contents(responses)
    # transform result to label
    results = []
    for idx, (c, (request, label)) in enumerate(zip(contents, ld)):
        c = c.strip().lower()
        if c.find('1') != -1:
            if c.find('2') != -1:
                print('???1')
                results.append((idx, c, request, label))
            else:
                results.append(1)
        elif c.find('2') != -1:
            if c.find('1') != -1:
                print('???2')
                results.append((idx, c, request, label))
            else:
                results.append(2)
        else:
            print('???3')
            results.append((idx, c, request, label))

    return responses, contents

