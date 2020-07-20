# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/25 12:39"
__doc__ = """ 快速排序"""

from random import randint,shuffle


def swap(A, i, j):
    A[i], A[j] = A[j], A[i]


def partition(A, low, high):
    """左右指针法：分割数组A, 左边元素小于tmp, 右边元素大于tmp"""
    # whistle = A[low]
    # rand_num = randint(low, high)
    tmp = A[low]        # 初始化基准值, 随机取值
    while low < high:
        # 倒序遍历
        while low < high and A[high] >= tmp:
            high -= 1
        A[low] = A[high]
        # 正序遍历
        while low < high and A[low] <= tmp:
            low += 1
        A[high] = A[low]
    # 写入基准值位置
    A[low] = tmp
    return low


def quicksort(A, p, r):
    """快速排序递归函数"""
    if p < r:
        q = partition(A, p, r)
        quicksort(A, p, q-1)
        quicksort(A, q+1, r)
    return A


if __name__ == '__main__':
    A = [0, -1, 3, 4, 5, -7, 2, 10, 1, 1, 87, 4, 21, 54, 2414]
    shuffle(A)
    p = 0
    r = len(A)-1
    # q = partition(A, p, r)
    # print(q, A)
    quicksort(A, p, r)
    print(A)




