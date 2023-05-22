import numpy as np
import matplotlib.pyplot as plt



def init_network(X):
    return init_network_v2(np.array([X]))

def init_network_v2(X):
    n, dim = X.shape
    matrix = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1, dim):
            count = 0
            for dd in range(n):
                attr = 1 if X[dd, i] == X[dd, j] else -1
                count += attr
            # print(count / n)
            matrix[i,j] = count / n
            matrix[j,i] = matrix[i,j]
    return matrix

# X: (16)
def energy(X, edge_matrix):
    count = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            attr = edge_matrix[i, j]
            count += attr * X[i] * X[j]
    return -count


def step(X, edge_matrix):
    res = X.copy()
    for i in range(len(X)):
        count = 0
        for j in range(len(X)):
            count += (edge_matrix[j, i] * X[j] if i != j else 0)
        res[i] = 1 if count > 0 else -1
    return res

def show(X, n = 4, name = 'dd', info = ''):
    plt.clf()
    plt.imshow(X.reshape(n,n))
    plt.colorbar()
    plt.xlabel(info)
    plt.savefig(f'{name}.png')


def random_X(n):
    res = np.ones(n, dtype = int)
    for i in range(n):
        res[i] = 1 if np.random.randint(2) == 1 else -1
    return res
    

def run():
    # X = np.array([1,1,1,1, 1,1,1,1, -1,-1,-1,-1, -1,-1,-1,-1]).reshape(4, 4)
    proto = np.array([1,-1, 1,-1,1,-1, 1,-1, 1,-1,1,-1,1,-1,1,-1])
    matrix = init_network(proto)
    show(proto, name = 'proto')
    # GO
    x = random_X(len(proto))
    ene_old = energy(x, matrix)
    show(x, name = 'dd', info = f'energy = {ene_old}')
    for i in range(20):
        x = step(x, matrix)
        ene = energy(x, matrix)
        if ene != ene_old:
            ene_old = ene
            show(x, name = f'dd{i}', info = f'energy = {ene}')
        else:
            break
        
def step_random(x, edge_matrix):
    res = x.copy()
    i = np.random.randint(16)
    count = sum([(edge_matrix[j, i] * x[j] if i != j else 0) for j in range(len(x))])
    res[i] = 1 if count > 0 else -1
    return res, res[i] != x[i]

def run2():
    X = np.array([[1,1,1,1, 1,1,1,1, -1,-1,-1,-1, -1,-1,-1,-1],
              [1,1,1,1, -1,-1,-1,-1, 1,1,1,1, -1,-1,-1,-1],
              [1,-1, 1,-1,1,-1, 1,-1, 1,-1,1,-1,1,-1,1,-1],
              [1,1,-1,-1,1,1, -1,-1, 1,1,-1, -1,1,1, -1,-1]])
    matrix = init_network_v2(X)
    # GO
    x = random_X(X.shape[1])
    ene_old = energy(x, matrix)
    show(x, name = 'dd', info = f'energy = {ene_old}')
    for i in range(20):
        x = step(x, matrix)
        ene = energy(x, matrix)
        if ene != ene_old:
            ene_old = ene
            show(x, name = f'dd{i}', info = f'energy = {ene}')
        else:
            break
    

def run3():
    X = np.array([[1,1,1,1, 1,1,1,1, -1,-1,-1,-1, -1,-1,-1,-1],
              [1,1,1,1, -1,-1,-1,-1, 1,1,1,1, -1,-1,-1,-1],
              [1,-1, 1,-1,1,-1, 1,-1, 1,-1,1,-1,1,-1,1,-1],
              [1,1,-1,-1,1,1, -1,-1, 1,1,-1, -1,1,1, -1,-1]])
    matrix = init_network_v2(X)
    # GO
    x = random_X(X.shape[1])
    ene_old = energy(x, matrix)
    show(x, name = 'dd', info = f'energy = {ene_old}')
    for i in range(100):
        x, sucess = step_random(x, matrix)
        if sucess:
            ene = energy(x, matrix)
            show(x, name = f'dd{i}', info = f'energy = {ene}')
