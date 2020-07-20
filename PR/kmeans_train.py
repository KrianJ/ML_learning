# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/23 14:53"
__doc__ = """ Kmeans聚类算法"""

import numpy as np
from PR.mat2mtx import dim3to2, augmentMatrix, Matrix2Image
from scipy import io as scio

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

train_imgs = scio.loadmat(r"training_set\mnist\train_images.mat")['train_images']       # 训练集
train_labels = scio.loadmat(r'training_set\mnist\train_labels.mat')['train_labels1'].ravel()    # 训练标签

# 设定训练参数
train_num = 8000
train_data = train_imgs[:,:,:train_num]     # 原始训练集(3维)
data = dim3to2(train_data)                  # 二维训练集
labels = train_labels[:train_num]

# 将原始训练集 -> 图片
# imgs = []
# for i in range(train_num):
#     im = Matrix2Image(train_data[:,:,i])
#     imgs.append(im)


# 简单构造聚类器并训练
estimator = KMeans(n_clusters=10)
estimator.fit(data)

# 预测原本训练集，判断聚类精度
count = 0
pred_labels = estimator.predict(data)
for i in range(train_num):
    if pred_labels[i] == labels[i]:
        count += 1
acc = count/train_num

# 输出聚类精度
print("聚类精度为:", acc)
print(estimator.score(data,labels))

