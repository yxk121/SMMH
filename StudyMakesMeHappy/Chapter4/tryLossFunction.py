import numpy as np


def mean_squared_error(y, t): # 均方误差
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7 # delta的作用是在取对数的时候不会出现log0的情况
    return -np.sum(t * np.log(y + delta))


