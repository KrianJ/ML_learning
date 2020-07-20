# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/2/28 15:21"
__doc__ = """ 感知机算法"""

import numpy as np
from scipy import io as scio
from ML.algorithm.perceptionLearn import perceptionLearn
from sklearn import preprocessing

# 读取数据
test_imgs = scio.loadmat(r'resources\mnist\test_images.mat')['test_images']
test_labels = scio.loadmat(r'resources\mnist\test_labels.mat')['test_labels1'][0]

# 设定数据量
train_num = 1200
test_num = 100
# 临时变量以及各个感知器参数
j = 0
alpha = 0.01    # 学习速率
epoch = 10      # 训练轮次
number = [8, 9]   # 要取的数字，由于是二分类，所以每次只能检验两个数字

# 提取number里的数字对应的样本数据，因为数据打乱过，不用shuffle直接取
data = np.empty(test_imgs.shape)
label = np.empty(test_labels.shape)
for i in range(10000):
    if test_labels[i] in number:
        data[:, :, j] = test_imgs[:, :, i]
        label[j] = test_labels[i]       # 取相应标签
        j += 1
    if j >= train_num + test_num:
        break

# 由于感知器输出结果仅为0、1，因此要将标签进行转换
# 由于没有进行规范化，后面更新权值w需要借助标签，因此标签需要置-1和1
for k in range(train_num + test_num):
    if label[k] == number[0]:
        label[k] = -1
    elif label[k] == number[1]:
        label[k] = 1

# 截取有效样本（非零）并标准化
data_ = []
for i in range(train_num+test_num):
    sample = data[:,:,i].flatten()      # 逐个取样本
    data_.append(list(sample))          # 样本加入data_
data_ = preprocessing.scale(np.array(data_))      # (train+num, 784)


test_data = np.hstack((data_[train_num:, :], np.ones((test_num, 1)))) # 对测试集进行增广变换
# test_data = data_[train_num+1:train_num+test_num, :]

# 训练权值
train_data = data_[:train_num, :]       # 取前train_num个为训练样本
train_label = label[:train_num]
w = perceptionLearn(train_data, train_label, alpha, epoch)      # 返回权重行向量w

# 测试（预测）
res = np.sign(np.dot(test_data, w))
print("测试集负例个数：", list(res).count(-1))
print("测试集正例个数：", list(res).count(1))
# res = [0] * test_num
# for k in range(test_num):
#     if np.sign(np.dot(test_data[k,:], w)):        # 元素为-1也是True
#        res[k] = 1
#     else:
#         res[k] = -1
# print(res.count(1))

# 输出预测的准确率
acc = 0
for sample in range(test_num):
    if res[sample] == label[train_num+sample]:
        acc = acc+1
print('精确度为: ', (acc/test_num)*100, '%');


