# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/26 15:43"
__doc__ = """ 文本分割"""

import jieba

# 自定义字典，加入一些特定人名和语料
"""杨过;过儿;过过(16)，把(11)，更响地(4), 并着(12), 科利亚(15)"""
jieba.add_word('杨过', tag='nr')
jieba.add_word('过儿', tag='nr')
jieba.add_word('过过', tag='vg')
jieba.add_word('科利亚', tag='nr')
jieba.add_word('更响地', tag='g')
jieba.add_word('并着', tag='v')
jieba.add_word('行行', tag='m')
# 人工干预，不符合语境
jieba.add_word('很快', freq=0)
jieba.add_word('一行行', freq=0)
jieba.add_word('行一行', freq=0)
jieba.add_word('行不行', freq=0)
jieba.add_word('行行行', freq=0)

import jieba

text = ['北京大学生在北京大学聚会。',
        '研究生命运都不错。\n研究生命科学很有趣。',
        '她冷射的手。',
        '稚气的声音更响地震动阿婆鼓膜。',
        '母亲信服地点点头。',
        '你反正是来得及赶到学校的！',
        '镇上那些老年人为什么来坐在教室里？',
        '她们谈到法国语言上来了。',
        '墙上装着一架电话的屋子',
        '对于为什么而读书有过多种不同的答案',
        "请把手放下去。\n自行车的把手坏了。",
        '手牵着手肩并着肩。',
        '粗大的雨点落下来了。',
        '她感到很快活。',
        '战争开始的时候，科利亚刚学数学。',
        '来到杨过曾经生活过的地方，小龙女动情地说：”我也想过过过儿过过的生活“',
        '人要是行，干一行行一行，一行行行行行；要是不行，干一行不行一行，一行不行行行不行']

for txt in text:
    # all_cut = jieba.cut(txt, cut_all=True, HMM=True)
    # print('全切分：', '/'.join(all_cut))
    specific_cut = jieba.cut(txt, cut_all=False, HMM=False)
    print(text.index(txt), '精确切分：', '/'.join(specific_cut))
    print("*********************")
    # break


