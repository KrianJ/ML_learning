# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/17 12:23"
__doc__ = """ """

import numpy as np


def data_initialize(A, B, pai, O):
    """数据初始化"""
    A = np.array(A).reshape((3, 3))
    B = np.array(B).reshape((3, 2))
    pai = np.array(pai).reshape((-1))
    O = np.array(O).reshape((-1)) - 1
    return A, B, pai, O


def alpha_compute(A, B, pai, seq: list):
    """递推计算前向变量alpha
    # t = 1时，初始化前向变量: alpha_1(i) = pai_i * b_i(O_1), 1 <= i <= N
    # t > 1时, 递推计算：alpha_t+1(j) = sum(alpha_t(i) * a_ij) * b_j(O_t+1)
    """
    T = len(seq)
    alpha = [0]                 # 初始化forward vector
    num_status = len(pai)       # 状态数

    # 计算t = i时的前向变量alpha_i
    for t in range(1, T+1):
        if t == 1:
            alpha_1 = np.array([pai[i] * B[i, seq[0]] for i in range(num_status)])
            alpha.append(alpha_1)
        else:
            alpha_t = np.array([np.sum(alpha[t-1] * A[:, j]) * B[j, seq[t-1]] for j in range(num_status)])
            alpha.append(alpha_t)
    return alpha


def forward_hmm(A, B, pai, seq: list):
    """
    HMM前向算法计算步骤
    :param A: 一步状态转移矩阵
    :param B: 观察序列概率矩阵
    :param pai: 初始分布
    :param seq: 给定观察序列
    :return: 给定观察序列seq出现的概率 & 前向向量alpha
    """
    # step1. initialize
    A, B, pai, seq = data_initialize(A, B, pai, seq)
    # step2. compute forward vector-alpha
    alpha = alpha_compute(A, B, pai, seq)
    # step3. probability of given sequence
    prob_seq = np.sum(alpha[-1])

    return prob_seq, alpha


if __name__ == '__main__':
    A = [0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333]
    B = [0.5, 0.5, 0.75, 0.25, 0.25, 0.75]
    pai = [0.333, 0.333, 0.333]
    O = [1, 2, 1, 2, 2]

    prob_O, alpha = forward_hmm(A, B, pai, O)
    print("出现序列 %s 的概率为：%s" % (O, prob_O))
    print("前向向量结果：", alpha)
