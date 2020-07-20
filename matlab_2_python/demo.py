# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2019/11/22 21:55"
__doc__ = """ """

import numpy as np
from scipy import io as scio
from scipy.linalg import eig


# fea_a = scio.loadmat('square_a.mat')
# data_a = fea_a['square_a']
# fea_b = scio.loadmat('square_b.mat')
# data_b = fea_b['square_b']
#
# nSmp_a = data_a.shape[0]
# nSmp_b = data_b.shape[0]
# bSqrt = 1

mtx = np.array(([2, 1], [1, 2]))
a, b = eig(mtx)      # a特征值 b特征向量
print(a, b)

# print(a.shape, b.shape)
# index_a = np.argsort(a)
# topk_a = index_a[:-4:-1]
# print(index_a, topk_a)
# tmp = np.array([1, 2, 3])
# topk_b = b[:, topk_a]
# print(topk_b.shape)
# print(type(a))
# topk_b = b[:, a[:-4:-1]]
# #
# print(b.shape)


# a = np.array([1, 2, 3, 4])
# print(a.shape)
# b = a.reshape(1, -1)
# print(b)
# sampledata = np.mean(data_a, 0).reshape(1, -1)
# print(sampledata.shape)
# data = data_a - np.tile(sampledata, (nSmp_a, 1))
# print(data.shape)

# aa = np.sum(data_a * data_a, axis=1)
# aa_T = aa.reshape(1, aa.shape[0])
# t = aa_T.transpose()
# bb = np.sum(data_b * data_b, axis=1)
# bb_T = bb.reshape(1, bb.shape[0])
# ab = data_a * data_b.T

# x = np.tile(aa, (1, nSmp_a))
# y = np.tile(aa_T, (nSmp_a, 1))
# z = 2*ab
# print(aa.shape, aa_T.shape, t.shape, x.shape, y.shape, z.shape)

#
# def line_tranpose(a):
#     """对列向量进行转置"""
#     try:
#         a.shape[1]
#     except IndexError:
#         return a.reshape(1, a.shape[0])
#
#
# c = line_tranpose(aa)
# print(aa, c)

# cc = np.zeros((1, 1))
# print(cc.any())
# print(np.tile(aa, (1, nSmp_b)).shape)
# print(np.tile(bb.T, (nSmp_a, 1)).shape)
# print((2 * ab).shape)
# if bSqrt:
#     D = np.sqrt(np.tile(aa, (1, nSmp_b)) + np.tile(bb.T, (nSmp_a, 1)) - 2 * ab)
#     D = np.real(D)
# else:
#     D = np.tile(aa, (1, nSmp_b)) + np.tile(bb.H, (nSmp_a, 1)) - 2 * ab
# u, s, vh = np.linalg.svd(data_a)
# print(vh[:, 0:20].shape)


