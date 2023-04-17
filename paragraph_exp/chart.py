import matplotlib.pyplot as plt
import numpy as np

AUXILIARY_LOSS_XS = [f'r{round(i * 0.1, 2)}' for i in range(16)]

def draw_line_chart(x, ys, legends, path = 'dd.png', colors = None, xlabel = None, ylabel = None):
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors is not None:
            plt.plot(x[:len(y)], y, colors[i], label = l)
        else:
            plt.plot(x[:len(y)], y, label = l)
    plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(path)
