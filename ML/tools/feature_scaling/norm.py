# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/1/9 15:51"
__doc__ = """ 对二维矩阵
        归一化normalization,
        标准化standlization"""


import numpy as np
from sklearn import preprocessing


def normalize(mtx, a=0, b=1):
    """
    normalization: remove dimensional effects
    range (a, b), default:(0, 1)
    formula: x* = a + k(x - min) / x* = b + k(x - max), k = (b - a)/(max - min)
    """
    data = mtx.ravel()          # 矩阵拉伸
    size = mtx.shape            # mtx's shape
    mx = np.max(data)           # max
    mn = np.min(data)           # min
    k = (b - a)/(mx - mn)       # step of (a, b)

    # 需要将矩阵拉伸，否则会报错：only size-1 arrays can be converted to Python scalars
    norm_data =  [a + k*(float(i) - mn) for i in data]
    return np.array(norm_data).reshape(size)


def standard(X):
    """
    z-score standardlization: require data has approximate Gaussian distribution, otherwise will be worse
    result: standard data obey N(0, 1)
    formula: X* = (X - X_mean) / sqrt(var)
    """
    scale_x = np.zeros(X.shape)
    mean_x = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    for j in range(X.shape[1]):
        if std[j] != 0:
            col = (X[:, j] - mean_x[j]) / std[j]
        else:
            col = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            scale_x[i, j] = col[i]
    return scale_x


# ------------------------Test Part-------------------------------
if __name__ == '__main__':
    arr = np.array([[1, 2, 3, 4],
                    [2, 2, 3, 1],
                    [3, 2, 5, 4]])
    a = preprocessing.scale(arr)
    # norm_arr = normalize(arr)
    standard_arr = standard(arr)
    # print(norm_arr)
    print(standard_arr)



