# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2019/12/1 15:52"
__doc__ = """ 将稀疏输入扩成满矩阵"""

import numpy as np
from scipy import sparse


def full(mtx):
    """sparse type matrix -> full matrix"""
    _sparse_mtx_type = ['bsr_matrix', 'coo_matrix', 'csc_matrix', 'csr_matrix',
                        'dia_matrix', 'dok_matrix', 'lil_matrix']

    mtx_type = str(type(mtx))
    for sparse_matrix in _sparse_mtx_type:
        if sparse_matrix in mtx_type:
            # if sparse_matrix == 'bsr_matrix':
            #     pass
            # elif sparse_matrix == 'coo_matrix':
            #     return mtx
            # elif sparse_matrix == 'csc_matrix':
            #     pass
            # elif sparse_matrix == 'csr_matrix':
            #     pass
            # elif sparse_matrix == 'dia_matrix':
            #     pass
            # elif sparse_matrix == 'dok_matrix':
            #     pass
            # elif sparse_matrix == 'lil_matrix':
            #     pass
            return mtx.toarray()
        else:
            return mtx


