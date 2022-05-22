import numpy as np


def bin2float(b):
    return int(b, 2)


def float2bin(f):
    return f'{f:064b}'


def map(i: float, x: list, y: list):
    return np.interp(i, x, y)


def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


def normalized_normal_dist(x, mean, sd, max_val):
    prob_density = (1/(np.sqrt(2*np.pi)*sd)) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density/max_val


def correlation_coefficient(D, E):
    p = np.corrcoef(D, E)
    return p[0, 1]
