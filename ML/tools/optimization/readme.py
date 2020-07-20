# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/1/14 20:47"
__doc__ = """ """


"""
All optimization algorithm:
: Hypothesis-------(假设拟合函数)
    h_theta(X) = theta.T ** X
: Cost Function----(代价函数)
    cost(h_theta(X), y)
: Model------------(模型)
    J(theta)
: Goal-------------(使J(theta)最小化的theta最优解)
    fit theta -> min J(theta)
:param theta: theta (n+1)*1 vector (theta_0 to theta_n, theta_0 = 1)(每个特征参数的系数向量theta)
:param X: character of sample(样本的特征参数向量，n个)
:param y: observations(观测值)

we can use different optimization algorithm in Linear Regression, Non-linear Regression, Classification and so on.
"""