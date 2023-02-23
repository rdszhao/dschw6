import yaml
import numpy as np

def load_config(path):
    with open(path, 'r') as file:
        x = yaml.load(file, Loader=yaml.SafeLoader)
    return x

def norma(x):
    norm = np.linalg.norm(x)
    return x / norm

def normalize(X):
    return np.apply_along_axis(norma, 1, X)

def calculateCorrect(y, t):
    z = np.argmax(y, axis=1)
    t2 = np.argmax(t, axis=1)
    return sum(z == t2)/len(t2)

def append_bias(X):
    row, _ = X.shape
    return np.c_[np.ones(row), X]

def load_data(path, norm=True):
    data = np.loadtxt(path, delimiter=',')
    label_idx = data.shape[1] - 1
    X = data[:, :label_idx]
    y = data[:, label_idx]
    y_flip = np.logical_not(y).astype(int)
    y = np.column_stack([y, y_flip])
    if norm:
        X = normalize(X)
    return X, y