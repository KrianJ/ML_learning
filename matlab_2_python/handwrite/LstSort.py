# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/24 15:22"
__doc__ = """ 插入排序"""


def insertsort(A:list)->list:
    pass


def _merge(A, p, q, r):
    """Split A into L, R, then merge L,R into an ordered new A"""
    # Split A into L, R
    L = A[p:q+1]
    R = A[q+1:r+1]
    # inserSort, i for L, j for R
    i, j = 0, 0
    for k in range(r):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
    if i != len(L):
        A[-1] = L[i]
    elif j != len(R):
        A[-1] = R[j]
    return A


def mergesort(A, p ,r)->list:
    if p < r:
        q = (p+r) // 2
        mergesort(A, p, q)
        mergesort(A, q+1, r)
        _merge(A, p, q, r)
    return A


if __name__ == '__main__':
    A = [1, 4, 6, 2, 10, 445, 0, 2, 56, 7, 87]
    p = 0
    r = len(A)-1
    q = (p+r) // 2
    print(_merge(A, p, q, r))
