# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/1/14 21:09"
__doc__ = """ 对数几率(logistic)回归
判别函数：G(x)=1/(1+exp(-W.T * X + b)) ---->  ln(y/1-y) = W.T * X + b
对数损失函数（交叉熵）：-1/m * sum(y*log f(x) + (1-y)*log(1-f(x)))
批量梯度下降更新：w = w - alpha * dw; b = b - alpha * db
dw = 1/m * (X.T * (y-f(x))); db = 1/m * sum(y-f(x))"""

import numpy as np


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def initialize(dims):
    """初始化参数"""
    w = np.zeros((dims,))
    b = 0
    return w, b


def logistic(X, y, W, b):
    """定义logistic主体模型，返回必要参数"""
    # 解决RuntimeWarning: divide by zero encountered in log问题
    # epsilon = 1e-5

    num_train, _ = X.shape[0], X.shape[1]
    # 矩阵形式logistic公式(特征维数的向量)
    f_x = sigmoid(np.dot(X, W) + b).reshape((num_train,))
    # 定义对数损失函数（交叉熵）: ln p(y|x) = y * log f_x + (1-y) * log(1-f_x)
    g_x = y * np.log(f_x) + (1-y) * np.log(1-f_x)
    cost = -1/num_train * np.sum(g_x)
    # 梯度下降更新：参数的偏导公式(基于g_x对w, b求偏导)
    dw = np.dot(X.T, (f_x - y)) / num_train
    db = np.sum(y - f_x) / num_train
    cost = np.squeeze(cost)                         # 删除单维度条目(shape[i]=1)

    return f_x, cost, dw, db


def logistic_train(X, y, alpha, epoch):
    # 初始化模型参数
    w, b = initialize(X.shape[1])
    cost_list = []

    # 迭代训练
    for i in range(epoch):
        # 计算当前次模型的计算结果，损失和参数梯度
        f, cost, dw, db = logistic(X, y, w, b)
        # 更新参数
        w += -alpha * dw
        b += -alpha * db

        # 记录损失(每100轮)
        if i % 100 == 0 or i == epoch-1:
            cost_list.append(cost)
            print("epoch %d cost %f" % (i, cost))

        # 保存参数
        params = {
            'w': w,
            'b': b
        }
        # 保存梯度
        grads = {
            'dw': dw,
            'db': db
        }
    return cost_list, params, grads


def Predict(X, params):
    """预测样本，并与现有样本比较"""
    w = params['w']
    b = params['b']
    y_pred = sigmoid(np.dot(X, w) + b)
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

