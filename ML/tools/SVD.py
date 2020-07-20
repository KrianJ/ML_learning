# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/1/9 15:35"
__doc__ = """ 对矩阵svd分解 """

import numpy as np
from numpy import linalg,sqrt



def mtx_svd(mtx):
    """M = UDV"""
    M = mtx
    M_T = mtx.T

    Z_v = np.dot(M_T, M)
    e_val, e_vecs = linalg.eig(Z_v)
    # 排序ATA的特征值特征向量
    sorted_eval_idx = np.argsort(e_val)[::-1]           # eval降序索引
    sorted_eval = [e_val[i] for i in sorted_eval_idx]   # 排序后的特征值
    v_sorted_evecs = e_vecs[:, sorted_eval_idx]           # 排序后的特征向量
    # 构造V矩阵，右奇异向量
    V = v_sorted_evecs


    # 构造奇异值对角阵D
    sin_val = [sqrt(eig) for eig in sorted_eval if eig!=0]
    D = np.diag(sin_val)


    Z_u = np.dot(M, M_T)
    val_u, vecs_u = linalg.eig(Z_u)
    # 排序AAT的特征值特征向量
    sorted_eval_idx = np.argsort(val_u)[::-1]           # eval降序索引
    u_sorted_evecs = vecs_u[:, sorted_eval_idx]         # 排序后的特征向量
    # 构造U矩阵，左奇异向量
    U = u_sorted_evecs

    return U, D, V

# -------------------------------Test Part-------------------------------
# if __name__ == '__main__':
#     mtx = np.array(([1, 2, 3],
#                     [2, 4, 7],
#                     [3, 7, 10],
#                     [4, 8, 5],
#                     [6, 9, 7]))
#     u,d,v = mtx_svd(mtx)
#     print(u,d,v)

