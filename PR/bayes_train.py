# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/8 23:53"
__doc__ = """ 贝叶斯分类器训练"""

import numpy as np
from sklearn import naive_bayes
from PR.mat2mtx import dim3to2, augmentMatrix
from scipy import io as scio
from time import time

train_imgs = scio.loadmat(r"training_set\mnist\train_images.mat")['train_images']       # 训练样本
train_labels = scio.loadmat(r'training_set\mnist\train_labels.mat')['train_labels1'].ravel()    # 训练标签
test_imgs = scio.loadmat(r'training_set\mnist\test_images.mat')['test_images']          # 测试样本
test_labels = scio.loadmat(r'training_set\mnist\test_labels.mat')['test_labels1'].ravel()       # 测试标签

# 对原始样本集降维并标准化
train_data = dim3to2(train_imgs)
train_data = augmentMatrix(train_data)     # 增广训练集
test_data = dim3to2(test_imgs)
test_data = augmentMatrix(test_data)       # 增广测试集
print(train_data.shape, test_data.shape)

"""
naive_bayes提供三种朴素贝叶斯: GaussianNB, MultinomialNB, BernouliNB
高斯朴素贝叶斯：适用于连续型数值，比如身高在160cm以下为一类，160-170cm为一个类，则划分不够细腻。
多项式朴素贝叶斯：常用于文本分类，特征是单词，值是单词出现的次数。
伯努利朴素贝叶斯：所用特征为全局特征，只是它计算的不是单词的数量，而是出现则为1，否则为0。也就是特征等权重。
"""
# def bayes_sample():
#     data = np.array([[1,3,3,6],
#                      [4,2,6,8],
#                      [3,7,9,1],
#                      [1,2,3,6],
#                      [2,1,3,6]])
#     labels = np.array([1,2,3,1,1])
#     clf = naive_bayes.MultinomialNB(alpha=2.0, fit_prior=True)
#     clf.fit(data, labels)
#     return clf

# 训练分类器
train_start = time()

classifier = naive_bayes.MultinomialNB(alpha=2, fit_prior=True, class_prior=None)    # 学习先验概率=各类样本数/样本总和
# classifier = naive_bayes.GaussianNB()
classifier.fit(train_data, train_labels)
# 数据量大时可以采用增量训练，第一次必须指定classes参数
# classifier.partial_fit(train_data, train_labels, classes=[0,1,2,3,4,5,6,7,8,9])

train_end = time()
print("training_runtime:", train_end - train_start)

# 1.了解MultinomialNB分类器属性
priors = classifier.class_prior             # 类先验概率
log_priors = classifier.class_log_prior_    # 类先验概率对数值
features = classifier.n_features_           # 特征数量
fea_log = classifier.feature_log_prob_      # 指定类的特征概率对数值
fea_count = classifier.feature_count_       # 各类别中各个特征出现的次数
classes = classifier.class_count_           # 各类别对应的样本数
linear_theta = classifier.coef_             # 将多项式模型映射成线性模型，值与fea_log_prob_相同
print(priors, features, fea_log, fea_count, classes, linear_theta)

# 2. 了解Gaussian分类器属性
# print(classifier.class_prior_)      # 类先验概率，也可以用priors
# print(classifier.theta_)            # 各个类标记在各个特征上的均值
# print(classifier.sigma_)            # 各个类标记在各个特征上的方差

# 分类器应用
params = classifier.get_params(deep=True)            # 获取分类器参数字典
class_proba = classifier.predict_proba(test_data)                  # 测试样本划分到各个类别的概率值
class_log_proba = classifier.predict_log_proba(test_data)          # 楼上的对数值
res = classifier.predict(test_data)                  # 测试集
print(res.shape)
score = classifier.score(test_data, test_labels)     # 测试精度
print("训练精度:", score)
for i in range(10):
    t = np.sum(res==i)
    r = np.sum(test_labels==i)
    err = abs(t-r)/r
    print('数字%d测试结果' % i, (i,t))
    print('数字%d真实结果' % i, (i,r))
    print('单个数字误差为:%.2f%%' % (err*100))




