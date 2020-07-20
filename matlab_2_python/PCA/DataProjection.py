# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2019/12/1 12:31"
__doc__ = """ 
X: D * N matrix of N data samples
r: dimension of the PCA projection, r=0 indicates no projection
Xp: r * N matrix of N projected data samples
"""
from numpy.linalg import svd


def data_projection(X, r):
    if r == 0:
        return X
    else:
        U, S, Vh = svd(X)       # return class tuple-(U, S, Vh)
        V = Vh.T
        Xp = U[:, 0:r].T
        Xp.dot(X)
        return Xp
