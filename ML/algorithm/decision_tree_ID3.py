# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/4/1 21:17"
__doc__ = """ ID3决策树实现
以信息增益(I)衡量每个条件属性，选择信息增益最大的条件属性作为当前根节点"""

from ML.algorithm.decision_tree_base import best_node_entropy


class ID3Tree:
    class Node:
        def __init__(self, name):
            self.name = name
            self.connections = {}

        def connect(self, label, node):
            self.connections[label] = node

    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node('Root')

    # print tree method
    def print_tree(self, node, tabs):
        print(tabs + node.name)
        for connection, child_node in node.connections.items():
            print(tabs + '\t' + '(' + connection + ')')
            self.print_tree(child_node, tabs + "\t\t")

    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)

    # construction_tree
    def construct(self, parent_node, parent_connection_label, input_data, columns):
        best_col, max_value, max_split = best_node_entropy(input_data[columns], self.label)
        if best_col is None:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return None

        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col not in [best_col]]
        # 递归构造决策树
        for splited_value, splited_data in max_split.items():
            self.construct(node, splited_value, splited_data, new_columns)
