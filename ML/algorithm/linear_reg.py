# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/1/14 21:08"
__doc__ = """ 线性回归
主题思路： 首先写出模型的主体和损失函数(loss-function)，以及损失函数的参数求偏导结果
再对参数进行初始化
最后写出基于梯度下降的更新过程
优化：通过交叉验证获得更稳健的参数值"""

import numpy as np


def linear_loss(X, y, w, b):
    """返回线性模型，损失函数，偏导w，偏导b"""
    num_train,_ = X.shape[0], X.shape[1]
    # 线性模型公式
    y_hat = np.dot(X,w) + b
    # 损失函数
    loss = 1/2 * np.sum((y_hat-y)**2) / num_train
    # w, b的偏导结果 dw = X(Xt*w - y), db = Xt*w - y
    dw = np.dot(X.T, (y_hat-y)) / num_train
    db = np.sum((y_hat-y)) / num_train
    return y_hat, loss, dw, db

def initialize(dims):
    """参数初始化"""
    w = np.zeros((dims,1))
    b = 0
    return w, b

def linear_train(X, y, alpha, epoch):
    """基于梯度下降(gradient descent)的线性模型训练"""
    w, b = initialize(X.shape[1])
    loss_list = []
    # gradient descent based training
    for i in range(epoch):
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        loss_list.append(loss)
        # update w,b based on gradient descent
        w += -alpha * dw
        b += -alpha * db
        # print each epoch and loss, epoch = 10000 * n based
        if i==0 or i == epoch-1:
            print('epoch %d loss %f' % (i, loss))
        # 迭代结束，保存参数
        if i == epoch - 1:
            # save params
            params = {
                'w': w,
                'b': b
            }
            # save gradient
            grads = {
                'dw': dw,
                'db': db
            }
    return loss_list, loss, params, grads




