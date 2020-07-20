# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/2/28 15:21"
__doc__ = """ 感知器算法训练得到的权重向量"""


import numpy as np


def perceptionLearn(x, y, alpha, maxEpoch):
    """
    :param x: 数据（行向量）
    :param y: 数据标签
    :param alpha: 学习速率（采用固定型）
    :param maxEpoch: 最大训练轮次
    :return: 训练得到的权重向量
    """
    rows, cols = x.shape
    x = np.hstack((x, np.ones((rows, 1))))    # 增广
    w = np.zeros(cols+1)       # 权重向量初始化
    for epoch in range(maxEpoch):
        flag = True                 # 标志为真则训练完毕
        for sample in range(rows):
            tmp = np.sign((x[sample, :]*w).sum())   # 标签的计算值
            if tmp != y[sample]:                    # 分类错误则更新权值
                flag = False
                # alpha = 1/(sample+1)                          # 变速学习速率
                w += alpha * y[sample] * (x[sample, :])         # 更新w，利用真实标签充当梯度方向的变量
                # w += alpha * (x[sample, :])
        if flag:
            break

    res = np.sign(np.dot(x, w))
    count = 0
    for i in range(len(res)):
        if res[i] == y[i]:
            count += 1
    print("训练后w对训练集的正确率：", count/rows)
    return w

