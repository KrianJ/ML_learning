# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/4/1 21:36"
__doc__ = """ """

import pandas as pd
import ML.algorithm.decision_tree_ID3 as ID3
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import pydotplus

df = pd.read_csv(r'D:\Pyproject\MLearning\ML\algorithm\resources\tennis.csv')

# 手写ID3算法实现
id3_tree = ID3.ID3Tree(df, 'play')      # 将play指定为决策属性
id3_tree.construct_tree()               # 构造决策树
id3_tree.print_tree(id3_tree.root, "")  # 打印决策树

# sklearn-tree模块实现
# iris = load_iris()
# clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
# clf = clf.fit(iris.data, iris.target)
# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=iris.feature_names,
#                                 class_names=iris.target_names,
#                                 filled=True,
#                                 rounded=True,
#                                 special_characters=True
#                                 )
# graph = graphviz.Source(dot_data)
