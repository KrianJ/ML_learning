# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/5 15:35"
__doc__ = """ 
1.将原始数据转化为SVM算法软件或包所能识别的数据格式；
2.将数据标准化；(防止样本中不同特征数值大小相差较大影响分类器性能)
3.不知使用什么核函数，考虑使用RBF；
4.利用交叉验证网格搜索寻找最优参数(C, γ)；（交叉验证防止过拟合，网格搜索在指定范围内寻找最优参数）
使用最优参数来训练模型；
5.测试。
"""

import numpy as np
from scipy import io as scio
from PR.mat2mtx import dim3to2
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# 读取数据集
train_imgs = scio.loadmat(r"training_set\mnist\train_images.mat")['train_images']       # 训练样本
train_labels = scio.loadmat(r'training_set\mnist\train_labels.mat')['train_labels1'].ravel()    # 训练标签
test_imgs = scio.loadmat(r'training_set\mnist\test_images.mat')['test_images']          # 测试样本
test_labels = scio.loadmat(r'training_set\mnist\test_labels.mat')['test_labels1'].ravel()       # 测试标签

# 对原始样本集降维并标准化
train_data = dim3to2(train_imgs)
train_data = np.hstack((train_data, np.ones((train_data.shape[0],1))))     # 增广训练集
test_data = dim3to2(test_imgs)
test_data = np.hstack((test_data, np.ones((test_data.shape[0],1))))       # 增广测试集

# 训练SVM分类器
"""C(惩罚参数)越大分类效果越好，可能会过拟合
gamma(rbf, poly, sigmond的核函数参数，默认auto)
kernel: rbf-高斯核; linear-线性核; poly-多项式; sigmond; precomputed
decision_function_shape: None;或者以下
    ovr一对多策略:一个类别与其他类别划分 
    ovo一对一策略:类别间两两划分，用二分类模仿多分类
degree: 多项式函数的维度"""
classifier = svm.SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo')
# 设定参数C, gamma范围
c_range = np.logspace(-5, 15, 11, base=2)
gamma_range = np.logspace(-9, 3, 13, base=2)
# 利用网格交叉验证搜索c,gamma范围，cv=3,3折交叉, 防止过拟合, 取最优参数
param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
grid = GridSearchCV(classifier, param_grid, cv=3, n_jobs=-1)
# 训练分类器
classifier.fit(train_data, train_labels)

# 计算分类准确率
print("训练集:", classifier.score(train_data, train_labels))
print("测试集:", classifier.score(test_data, test_labels))

# 查看决策函数-样本到分类超平面的
# print("决策函数:", classifier.decision_function(train_data))

"""
linear:
训练集: 0.9833666666666666
测试集: 0.926
rbf:
训练集: 0.98645
测试集: 0.966
"""




