import numpy as np

# y = [[1, 1, 1], [1, 1, 1]]
# x = [[1, -1, 1], [-1,-1,-1]]
# d = [1, 3]
def hamming(x, y):
    d = []
    for xx, yy in zip(x, y):
        dd = 0
        for xxx, yyy in zip(xx, yy):
            if xxx == 1 and yyy != 1:
                dd += 1
            elif yyy == 1 and xxx != 1:
                dd += 1
        d.append(dd)
    return d


X = np.array([[1,1,1,1, 1,1,1,1, -1,-1,-1,-1, -1,-1,-1,-1],
              [1,1,1,1, -1,-1,-1,-1, 1,1,1,1, -1,-1,-1,-1],
              [1,-1, 1,-1,1,-1, 1,-1, 1,-1,1,-1,1,-1,1,-1],
              [1,1,-1,-1,1,1, -1,-1, 1,1,-1, -1,1,1, -1,-1]])

N = np.shape(X)[1]
n = 4
b = np.zeros((1, N))
b = np.sum(X, axis = 0) / n
W = (X.T@X) / n - np.eye(N)


