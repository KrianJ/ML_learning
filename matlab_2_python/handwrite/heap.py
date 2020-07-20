# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/24 21:28"
__doc__ = """ 堆排序"""


def swap(A, i, j):
    """交换数组A[i]和A[j]"""
    A[i], A[j] = A[j], A[i]
    return None


def max_heapify(A:list, i):
    """将以i为根的子树改造成大根堆"""
    h_size = len(A)-1
    l = 2*i          # 左孩子
    r = 2*i + 1      # 右孩子
    largest = 0
    if l <= h_size and A[l] > A[i]:     # 左孩子最大
        largest = l
    else:
        largest = i
    if r <= h_size:     # 右孩子最大
        if A[r] > A[l] and A[r] > A[i]:
            largest = r
        elif A[l] > A[i]:
            largest = l
        else:
            largest = i
    # 做节点交换，则继续维护交换后的孩子节点
    if largest != i:
        swap(A, i, largest)
        max_heapify(A, largest)
    return None


def build_max_heap(A):
    """给定数组A, 建立大根堆"""
    heap_size = len(A)
    for i in range(heap_size//2, 0, -1):
        max_heapify(A, i)
    return None


def heap_sort(A:list):
    """给定大根堆A, 转换成有序数组(从大到小)"""
    sort_A = []
    length = len(A)-1
    for i in range(length, 0, -1):
        swap(A, 1, i)
        sort_A.append(A.pop(-1))
        max_heapify(A, 1)
    return sort_A


if __name__ == '__main__':
    A = [0, 5, 1, 3, 8, 10, 17, 106, 31, 78]
    build_max_heap(A)
    res = heap_sort(A)
    print(res)
