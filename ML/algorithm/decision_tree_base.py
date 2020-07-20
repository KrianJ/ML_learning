# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/4/1 15:47"
__doc__ = """ 决策树 基本要素实现"""

from math import log
import pandas as pd


def entropy(attr):
    """
    计算条件属性attr的信息熵
    :param attr: 条件属性
    :return: 信息熵
    """
    # 计算attr各个取值的概率
    probs = [attr.count(i)/len(attr) for i in set(attr)]
    # 计算信息熵
    entropy = -sum([prob * log(prob, 2) for prob in probs])
    return entropy


def split_dataframe(data, col):
    """根据属性col划分数据集"""
    # col所有取值
    values = data[col].unique()
    # 根据col的不同取值建立字典{value: dataframe object}
    result_dict = {elem: pd.DataFrame for elem in values}
    # split dataframe based on column value
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key]        # col列值等于key的所有行，即划分出的数据集
    return result_dict


def best_node_entropy(df, label):
    """
    choose the best (max I) attribute as the current root
    :param df: DataFrame
    :param label: df的结果属性（样本标签）
    :return:
        col(current_root): who has the max information gain
        max_info_gain: max information gain of col
        best_split: the best split of dataset based on (col, max_info_gain)
    """
    # 无条件属性干扰下的原始信息熵
    entropy_a = entropy(df[label].tolist())
    # 获取所有条件属性
    other_cols = [col for col in df.columns if col not in [label]]
    # 计算条件属性的信息增益
    max_info_gain, best_col = -1, None
    best_split = None

    # 根据col划分数据集，计算信息增益
    if len(other_cols) == 0:
        return None, 0, None
    for col in other_cols:
        split_subsets = split_dataframe(df, col)
        col_entropy = 0
        # 计算各个子集的信息熵,加权计算条件属性信息熵
        for subset_col, subset in split_subsets.items():
            # print(subset)
            sub_entropy = entropy(subset[label].tolist())       # 条件属性取值：subset_col, 对应subset的信息熵
            col_entropy += len(subset) / len(df) * sub_entropy  # 该条件属性的信息熵：加权和
        # 条件属性(col)的信息增益
        info_gain = entropy_a - col_entropy
        # 判断
        if info_gain > max_info_gain:
            max_info_gain, best_col = info_gain, col        # 最大信息增益，条件属性名
            best_split = split_subsets                      # 对应数据集划分

    return col, max_info_gain, best_split

