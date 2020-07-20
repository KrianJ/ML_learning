# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/5 15:41"
__doc__ = """ 将mat转换成二维矩阵"""

import numpy as np
from numpy import column_stack, row_stack
from sklearn import preprocessing
from PIL import Image


def dim3to2(data, direction=2):
    """
    :param data: 三维矩阵
    :param direction: 解构方向
    :return: 标准化后的二维矩阵
    """
    n = data.shape[direction]       # 解构三维矩阵方向
    data_ = []
    for page in range(n):
        if direction == 2:
            sample = list(data[:, :, page].flatten())
        elif direction == 1:
            sample = list(data[:, page, :].flatten())
        elif direction == 0:
            sample = list(data[page, :, :].flatten())
        else:
            return None
        data_.append(sample)
    data_ = preprocessing.scale(np.array(data_))
    return data_


def augmentMatrix(A, aug=1, axis=1):
    """
    增广矩阵A
    :param A: 矩阵A
    :param aug: 增广参数
    :param axis: 增广方向，1: 列增广，0：行增广
    :return: 增广矩阵A*
    """
    if axis == 1:
        rows = A.shape[0]
        if aug == 1:
            aug_col = np.ones(rows)
            return column_stack((A, aug_col))
        else:
            aug_col = np.array([aug]*rows)
            return column_stack((A, aug_col))
    elif axis == 0:
        cols = A.shape[1]
        if aug == 1:
            aug_col = np.ones(cols)
            return row_stack((A, aug_col))
        else:
            aug_col = np.array([aug]*cols)
            return row_stack((A, aug_col))


def Image2Matrix(img):
    """图片转矩阵"""
    pass


def Matrix2Image(mtx):
    """矩阵转图片"""
    new_im = Image.fromarray(mtx.astype(np.uint8))
    return new_im


if __name__ == '__main__':
    from PIL import Image
    # A = np.array(([1,2,3],
    #              [4,5,6],
    #              [7,8,9]))
    # print(augmentMatrix(A, axis=0))
    path = r'training_set\handWritingNum\2\10.bmp'
    img = np.array(Image.open(path))
    data = dim3to2(img, direction=2)
    print(data.shape)



