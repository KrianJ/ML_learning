# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/29 15:22"
__doc__ = """ 对数几率回归训练, 利用sklearn生成模拟的二分类数据集"""

from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from ML.algorithm.logistic import logistic_train, Predict
from sklearn.preprocessing import scale


def load_data(offset):
    """加载数据集
    iris: 3类 150*4"""
    iris = load_iris()
    data = iris.data
    labels = iris.target
    # 取两类样本
    two_class_length = len([i for i in range(len(labels)) if labels[i] != 2])
    data, labels = data[:two_class_length, :], labels[:two_class_length]

    # 划分数据集
    X, y = shuffle(scale(data), labels)
    X_train, y_train = X[:offset, :], y[:offset]
    X_test, y_test = X[offset:, :], y[offset:]
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # 初始化数据集
    X_train, y_train, X_test, y_test = load_data(80)
    # 训练
    loss_lst, params, _ = logistic_train(X_train, y_train, 0.05, 500)
    # 测试
    y_pred = Predict(X_test, params)
    res = [i for i in range(X_test.shape[0]) if y_pred[i] == y_test[i]]
    # 精度
    acc = len(res)/X_test.shape[0]
    print("精度为：", acc)















