# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2019/12/1 15:05"
__doc__ = """ 利用零均值化后矩阵的协方差矩阵进行PCA维数约简
input: 
    data: data matrix, each row vector represents a data point
    options.ReduceDim: the dimensionality of the reduced subspace(0: all dimensions kept)
output:
    eigvector: whose column is an embedding function(嵌入函数), explains that
               y = x * eigvector, y represents reduced x
    eigvalue: sorted eigvalue of PCA eigen-problem(PCA问题的分类特征值)
"""

import numpy as np
from scipy.linalg import eig
from matlab_2_python.BasicFunctions.sparse_basic_func import full
from time import time


def pac(data, options=dict):
    start_time = time()

    Reducedim = 0       # 样本约简后的维数
    """
    if isfield(options,'ReducedDim')
        ReducedDim = options.ReducedDim;
    end
    """
    # nSmp: 样本数
    # nfea: 样本维数
    nSmp, nFea = data.shape[0], data.shape[1]
    if Reducedim > nFea or Reducedim <= 0:
        Reducedim = nFea
    # data是稀疏输入则扩成满矩阵
    data = full(data)

    """
            part0: 对原始数据集零均值化
    """
    # 压缩行，求各列(样本特征参数)平均值，返回行向量
    sampleMean = np.mean(data, 0).reshape(1, -1)
    # 将matrix data零均值化data = data - data_mean
    data = data - np.tile(sampleMean, (nSmp, 1))

    if float(nFea/nSmp) > 1.0713:
        """compute the eigvector of dot(A, A.T), instead of dot(A.T, A)
        then convert them back to the eigenvectors of dot(A.T, A)"""
        # 计算零均值化后的matrix data的协方差矩阵d_data
        d_data = data.dot(data.T)
        d_data = np.maximum(d_data, d_data.T)
        # 协方差矩阵维数
        dimMatrix = d_data.shape[1]
        """
        part1: 提取数据
        topk_evals, topk_evecs: 前k大的特征值和对应特征向量
        mtx_evals: 特征值对角阵
        """
        if dimMatrix > 1000 and Reducedim < dimMatrix/10:
            options.update({'disp': 0})
            # 计算d_data的特征值和特征向量
            evals, evecs = eig(d_data)
            # 前k大个特征值的索引值(k = ReduceDim)
            topk_evals_index = np.argsort(evals)[:-Reducedim-1:-1]
            # 取对应的k个特征向量的矩阵
            topk_evecs = evecs[:, topk_evals_index]
            # 特征值对角阵
            mtx_evals = np.diag(evals)
        else:
            # 不取topk, 对特征值特征向量排个序
            evals, evecs = eig(d_data)
            mtx_evals = np.diag(evals)
            # 降序排序特征值
            sorted_evals = sorted(evals, reverse=True)
            # 对应的特征向量构成的矩阵
            sorted_evals_index = np.argsort(evals)
            sorted_evecs = evecs[:, sorted_evals_index]
        """
        part2: 1. d-dimension data = dot(topk_evecs.T, data)
               2. 再对d-dimension data归一化
        """
        """matlab code:
            clear ddata;
            maxEigValue = max(abs(eigvalue));
            eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
            eigvalue (eigIdx) = [];
            eigvector (:,eigIdx) = [];  
        """
    else:
        """这里源代码和if块里一样，需要重新修改"""
        d_data = np.dot(data.T, data)
        d_data = np.maximum(d_data, d_data.T)

        dimMatrix = d_data.shape[1]
        if dimMatrix > 1000 and Reducedim < dimMatrix/10:
            pass

    if Reducedim < len(evals):
        evals = evals[: Reducedim]
        evecs = evecs[:, :Reducedim]

    """
    if isfield(options,'PCARatio')
        sumEig = sum(eigvalue);
        sumEig = sumEig*options.PCARatio;
        sumNow = 0;
        for idx = 1:length(eigvalue)
            sumNow = sumNow + eigvalue(idx);
            if sumNow >= sumEig
                break;
            end
        end
        eigvector = eigvector(:,1:idx);
    end
    """

    reduced_data = np.dot(data.T, topk_evecs)
    """这里再加上对reduced_data归一化"""
    runtime = time() - start_time

    return reduced_data, runtime







