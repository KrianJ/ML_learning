# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/27 21:50"
__doc__ = """ 利用sklearn diabetes进行简单训练
diabetes 是一个关于糖尿病的数据集， 该数据集包括442个病人的生理数据及一年以后的病情发展情况。   
数据集中的特征值总共10项, 如下:  
    # 年龄  
    # 性别  
    #体质指数  
    #血压  
    #s1,s2,s3,s4,s4,s6  (六种血清的化验数据)  
    #每个特征都做了标准化处理，0均值1方差"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
from ML.algorithm.linear_reg import linear_train


diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target

# 打乱数据
X, y = shuffle(data, target)
X.astype(np.float32)
# 训练集，测试集划分
offset = int(X.shape[0] * 0.9)      # 90%的划分比
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
# 将标签转换成一列
y_train, y_test = y_train.reshape((-1, 1)), y_test.reshape((-1, 1))

# 训练
loss_lst, loss, params, grads = linear_train(X_train, y_train, 0.05, 100000)
print(params, grads)

# 预测
def Predict(X):
    w = params['w']
    b = params['b']
    pred = np.dot(X, w) + b
    return pred

y_pred = Predict(X_test)
# print("预测值：",y_pred.reshape(-1))
# print("真实值：", y_test.reshape(-1))
# print("预测误差：", y_pred.reshape(-1) - y_test.reshape(-1))

# 预测-真值曲线
import matplotlib.pyplot as plt
f = X_test.dot(params['w']) + params['b']     # 所有预测值

plt.scatter(range(X_test.shape[0]), y_test, c='red')     # 真值散点图
# plt.scatter(range(X_test.shape[0]), f, c='blue')          # 测试值离散图
# plt.plot(y_test, color='red')    # 真值折线图
plt.plot(f, color='blue')        # 测试值折线图
plt.xlabel(r'X')
plt.ylabel(r'y')
plt.show()

# 损失下降曲线
plt.plot(loss_lst, color='blue')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
