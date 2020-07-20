# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2019/11/22 16:26"
__doc__ = """ This routine is used to construct a weighted matrix which is hoped 
to characterize the structure of a given data set: 这个程序是用来构建一个加权矩阵，希望
描述给定数据集的结构"""

import numpy as np
import scipy.io as scio
from time import time
from random import randint

# X = np.array()
# Y = np.array()


def constructM(X, Y, flag):
    """
    input: X, Y-<class 'ndarray'>, flag
    X, Y: 行样本列特征
    ouput: M, time
    M: 描述给定数据集的加权矩阵
    """
    start_time = time()
    switch_flag = {
        1:  'Gaussian kernel',
        2: 'method_SC_LRR'
    }
    # 矩阵X, Y的行(D)列(Y)数
    Dy, Ny = Y.shape[0], Y.shape[1]
    Dx, Nx = X.shape[0], X.shape[1]
    if Dy != Dx:
        return 'Fail!'

    M = np.zeros((Ny, Nx))
    # flag: 1-高斯核
    if switch_flag[flag] == 'Gaussian kernel':
        s = 0
        for i in range(Ny):
            for j in range(Nx):
                s = sum((Y[:, i] - X[:, j]) ** 2) + s
        # average distance between pairwise points from X and Y
        s = s/(Ny * Nx)
        for i in range(Ny):
            for j in range(Nx):
                M[i:j] = 1 - np.exp(-sum((Y[:, i]-X[:, j]) ** 2)/s)
    # flag: 2-SC_LRR
    elif switch_flag[flag] == 'method_SC_LRR':
        s = 0
        for i in range(Ny):
            for j in range(Nx):
                s = 1 - np.abs(Y[:, i].T.dot(X[:, j])) + s
        s = s/(Ny * Nx)
        for i in range(Ny):
            for j in range(Nx):
                if i == j:
                    M[i:j] = 1
                M[i:j] = 1 - np.exp(-(1-np.abs(Y[:, i].T.dot(X[:, j]))) / s)

    runtime = time() - start_time

    return [M, runtime]


def main():
    # 1. data initialize
    X = scio.loadmat('train_data/train1.mat')       # <class 'dict'>
    X_data = X['train1'][0:5958]                    # <class 'ndarray'>
    Y = scio.loadmat('train_data/train2.mat')
    Y_data = Y['train2']
    flag = randint(1, 2)
    # 2. calculate
    res = constructM(X_data, Y_data, flag=flag)
    M = res[0]
    runtime = res[1]

    print(M.shape)
    print("matrixM:", M, '\n', "total time:", runtime)


if __name__ == '__main__':
    main()
