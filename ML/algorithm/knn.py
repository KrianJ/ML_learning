# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/31 15:19"
__doc__ = """ k近邻分类算法
component:
    Train_data: mnist; 
    Similarity function; Euclid distance
    Param:k
    Decision rule:"""

import numpy as np
from scipy.io import loadmat
from PR.mat2mtx import dim3to2
from collections import Counter
from time import time
import os
import tensorflow as tf
from memory_profiler import profile


def eu_distance(X, X_train):
    """
    L2 norm, 即euclid distance。
    :param X: 目标数据 i*j
    :param X_train: 训练数据 h*j
    :return: 样本集X到样本集X_train的距离（矩阵）i*h
    """
    if len(X.shape) != 1:
        rows, cols = X.shape[0], X_train.shape[0]
        eu_dis = np.zeros((rows, cols))        # 初始化距离矩阵

        # 计算X每个样本到X_train每个样本的距离，按行填充
        for i in range(rows):
            i_X = X[i, :]
            i_dis = [np.sqrt(np.sum((i_X - X_train[h, :])**2)) for h in range(X_train.shape[0])]    # 1*h
            eu_dis[i, :] = i_dis
        return eu_dis
    else:
        single_dis = [np.sqrt(np.sum((X - X_train[h, :])**2)) for h in range(X_train.shape[0])]    # 1*h
        return single_dis


def load_data():
    mat_path = [r'D:\Pyproject\MLearning\ML\datasets\mnist\test_images.mat',
                r'D:\Pyproject\MLearning\ML\datasets\mnist\test_labels.mat',
                r'D:\Pyproject\MLearning\ML\datasets\mnist\train_images.mat',
                r'D:\Pyproject\MLearning\ML\datasets\mnist\train_labels.mat']
    data = loadmat(mat_path[2])['train_images']
    labels = loadmat(mat_path[3])['train_labels1']
    # 处理数据规模
    x = dim3to2(data)
    y = labels.reshape((-1,))
    return x, y


def Predict_labels(y_train, dist, k=1):
    """采用多数表决的分类规则, 预测测试集样本属于已知训练集中哪一类
    :param y_train: 训练集标签
    :param dist: 距离矩阵
    :param k: 近邻样本数
    :return y_pred[i]：第i个样本按照决策规则投票归属的类标签"""
    num_test = dist.shape[0]
    y_pred = np.zeros(num_test)
    # 对每个测试样本取其k近邻
    for i in range(num_test):
        # 对dist第i行排序，提取对应下标的y_train,取前k个近邻
        labels = y_train[np.argsort(dist[i, :])]            # dist[i, :]第i个样本和所有样本的距离
        closest_y = labels[:k]                              # 取k近邻标签
        # 定义一个Counter类，k近邻中出现最多的标签即为样本i的类标签
        c = Counter(closest_y)
        y_pred[i] = c.most_common(1)[0][0]                  # Counter对象中出现次数最多的标签名
    return y_pred


# @profile(precision=4)
def knn_cross_validation(X, y):
    """交叉验证取最佳的k值"""
    # 5折交叉验证
    num_folds = 5
    k_choice = [1, 3, 5, 8, 10, 20]
    k_acc = {}
    # 将样本，标签分成5份
    X_folds = np.array_split(X, num_folds)
    y_folds = np.array_split(y, num_folds)

    # 迭代k的不同取值情况
    for k in k_choice:
        # 每个k对应num_folds次训练，一共有len(k_choice) * num_folds次结果
        for fold in range(num_folds):
            # 划分训练集和测试集
            X_test = X_folds[fold]
            y_test = y_folds[fold]
            X_train = np.concatenate(X_folds[:fold] + X_folds[fold+1:])
            y_train = np.concatenate(y_folds[:fold] + y_folds[fold+1:])
            # 计算距离
            _dis = eu_distance(X_test, X_train)
            # 预测 & 分析精度
            y_pred = Predict_labels(y_train=y_train, dist=_dis, k=k).reshape((-1, ))
            num_correct = np.sum(y_pred == y_test)
            num_test = X_test.shape[0]
            acc = float(num_correct) / num_test
            # 写入k_acc字典中
            k_acc[k] = k_acc.get(k, []) + [acc]
    return k_acc


if __name__ == '__main__':
    # 调用GPU
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # # 设置GPU用量
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    # session = tf.Session(config=config)
    
    # 数据处理
    start_time = time()
    X, y = load_data()
    # knn_cross_validation
    X = X[:2000, :]
    y = y[:2000]
    k_acc = knn_cross_validation(X, y)
    for index, k in enumerate(k_acc):
        print("采用5折交叉验证，k取值 %d时的精度为: " % k, k_acc[k], "平均精度：", np.mean(k_acc[k]))
    run_time = time() - start_time
    print("运行时间：", run_time)

    # x_train = X[:500, :]
    # y_train = y[:500]
    # x_test = X[450:500, :]
    # y_test = y[450:500]

    # ** ** ** ** ** ** ** ** Function->Predict_labels测试代码 ** ** ** ** ** ** ** ** ** ** ** ** *
    # # 测试集对训练集的距离矩阵
    # dist = eu_distance(x_test, x_train)
    # # 按照对训练集中k近邻规则，预测测试集样本属于哪一类
    # y_pred = Predict_labels(y_train=y_train, dist=dist, k=1).reshape((-1, ))
    # # 结果统计
    # num_correct = np.sum(y_test == y_pred)
    # acc = float(num_correct) / x_test.shape[0]
    # print(num_correct, acc)

    # ** ** ** ** ** ** ** ** Function->eu_distance测试代码 ** ** ** ** ** ** ** ** ** ** ** ** *
    # dis = eu_distance(x_test, x_train)
    # plt.imshow(dis, interpolation='none')
    # plt.show()



