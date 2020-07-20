# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/24 16:13"
__doc__ = """ 最大子数组"""


def find_max_cross_subarray(A, low, mid, high):
    # 左侧最大子数组
    left_sum = -1
    sum = 0
    for i in range(mid, low,  -1):
        sum += A[i]
        if sum > left_sum:
            left_sum = sum
            max_left = i
    # 右侧最大子数组
    right_sum = -1
    sum = 0
    for j in range(mid+1, high):
        sum += A[j]
        if sum > right_sum:
            right_sum = sum
            max_right = j

    return max_left, max_right, left_sum+right_sum


def find_max_subarray(A, low, high):
    if high == low:
        return low, high, A[low]
    else:
        # 递归式
        mid = (low+high) // 2
        l_low, l_high, l_sum = find_max_subarray(A, low, mid)
        r_low, r_high, r_sum = find_max_subarray(A, mid+1, high)
        c_low, c_high, c_sum = find_max_cross_subarray(A, low, mid, high)
        # 比较最大子数组
        max_sum = max(l_sum, r_sum, c_sum)
        if l_sum == max_sum:
            return l_low, l_high, l_sum
        elif r_sum == max_sum:
            return r_low, r_high, r_sum
        else:
            return c_low, c_high, c_sum


if __name__ == '__main__':
    A = [0, -1, 3, 4, 5, -7, 2]
    low = 0
    high = len(A)
    print(find_max_subarray(A, low, high))

