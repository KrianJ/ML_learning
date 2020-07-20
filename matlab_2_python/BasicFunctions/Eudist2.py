# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2019/11/27 10:50"
__doc__ = """ 欧式距离矩阵
 D = EuDist(fea_a,fea_b)
fea_a: nSample_a * nFeature
fea_b: nSample_b * nFeature
D:     nSample_a * nSample_a
   or  nSample_a * nSample_b
"""

import numpy as np
from scipy import io as scio


def line_tranpose(a):
    """ python列向量的转置还是列向量
        所以要对列向量reshape实现转置"""
    try:
        a.shape[1]
    except IndexError:
        return a.reshape(1, a.shape[0])


def Eudist2(fea_a, fea_b=np.zeros((1, 1)), bSqrt=1):
    """require 2 n*n data matrix
    return Eudist mtxD:
    1. Eudist2(fea_a): distance in fea_a's samples
    2. Eudist2(fea_a, fea_b): distance of fea_a & fea_b"""
    if not fea_b.any():
        nSmp = fea_a.shape[0]       # 样本数

        aa = np.sum(fea_a * fea_a, axis=1)  # 压缩成列向量
        aa_T = line_tranpose(aa)        # aa的转置
        ab = np.dot(fea_a, fea_a.T)     # 转换成实对称阵

        if bSqrt:
            D = np.sqrt(np.tile(aa, (nSmp, 1)) + np.tile(aa_T, (nSmp, 1)) - 2*ab)
            D = np.real(D)
        else:
            D = np.tile(aa, (nSmp, 1)) + np.tile(aa_T, (nSmp, 1)) - 2*ab

        D = np.maximum(D, D.T)
        D = D - np.diag(np.diag(D))

    else:
        nSmp_a = fea_a.shape[0]
        nSmp_b = fea_b.shape[0]

        aa = np.sum(fea_a * fea_a, axis=1)
        bb = np.sum(fea_b * fea_b, axis=1)
        bb_T = line_tranpose(bb)
        ab = fea_a * fea_b.T

        if bSqrt:
            """ndarray对象中列向量是以[]形式, 行向量是以[[]]形式
            但是tile两个不同的向量为什么rep是一样的"""
            D = np.sqrt(np.tile(aa, (nSmp_b, 1)) + np.tile(bb_T, (nSmp_a, 1)) - 2*ab)
            D = np.real(D)
        else:
            D = np.tile(aa, (nSmp_b, 1)) + np.tile(bb_T, (nSmp_a, 1)) - 2*ab

    return np.abs(D)


def main():
    fea_a = scio.loadmat('train_data/square_a.mat')
    data_a = fea_a['square_a']
    fea_b = scio.loadmat('train_data/square_b.mat')
    data_b = fea_b['square_b']
    D = Eudist2(data_a, data_b)
    # D= Eudist2(data_a)
    print(D)


if __name__ == '__main__':
    main()
