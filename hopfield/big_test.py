from PIL import Image
import numpy as np
from taku_toy import *

def get_img():
    # source
    im = Image.open('homo.jpeg')
    pixelMap = im.load()
    # new
    img = np.zeros((64, 64), dtype = int)
    neck = 340
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 1 if sum(pixelMap[j,i]) > neck else -1
    return img

# init network

def run3():
    img = get_img()
    proto = img.reshape(-1)
    matrix = init_network(proto)
    # GO
    x = random_X(len(proto))
    ene_old = energy(x, matrix)
    show(x, n = 64, name = 'dd', info = f'energy = {ene_old}')
    for i in range(1000):
        x, sucess = step_random(x, matrix)
        if sucess:
            ene = energy(x, matrix)
            show(x, n = 64, name = f'dd{i}', info = f'energy = {ene}')


