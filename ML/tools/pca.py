# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/1/10 19:43"
__doc__ = """ 几种PCA算法粗略实现"""


import numpy as np
from numpy import linalg


def pca(X, d):
    """classic PCA: deal with Gaussian noise
    :param X: vector samples(row)
    :param d: target dimension
    :return: reduced_X: reduced vector samples
    """
    # 1. zero-equalization
    size = X.shape[0]
    mean_x = np.mean(X)
    new_X = np.array([x - mean_x for x in X])     # 零均值化
    # 2.Cov(X)
    tmp_a = new_X.reshape(size, 1)
    tmp_b = new_X.reshape(1, size)
    cov_X = np.dot(tmp_a, tmp_b)            # 协方差矩阵
    # 3.eig_value & eig_vectors
    e_val, e_vecs = linalg.eig(cov_X)
    eval_idx = np.argsort(e_val)[::-1]
    # 4.取出前d大的特征值对应特征向量
    sorted_eval = [e_val[i] for i in eval_idx]
    sorted_vec = e_vecs[:, eval_idx]
    d_vec = sorted_vec[:, :d]               # 每个特征向量对应一个主成分方向
    # 5. X降维结果(投影结果)
    reduced_X = np.dot(d_vec.T, X)
    # i = np.dot(d_vec.T, d_vec)

    return reduced_X


def rpca(X, d):
    """
    Robust PCA: X = D + E, deal with sharp peak noise
    :param X: vector sample(row)
    :param d: reduced dimension
    :return: D:low-rank matrix, E: sparse matrix
    """
    # 1.利用快速ALM对X进行RPCA分解，L(mn*1,low rank) + E(sparse)

    # 2.reshape L, E
    pass



# -----------------------Test Part-------------------------
if __name__ == '__main__':
    # a = np.array(([1, 2, 3],
    #               [2, 5, 6],
    #               [3, 6, 9]))
    a = np.array(([1,2,3,4,5,6,7,8]))
    a = pca(a,4)
    print(a)



