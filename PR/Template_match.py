# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/2/25 19:39"
__doc__ = """ 车牌模板匹配"""


import numpy as np
from PIL import Image

def imread(filename):
    """读取文件路径，返回拉伸后的一维行向量"""
    IM = Image.open(filename).convert('L').resize((28, 28))  # 模式L: 灰度图像 0~255,统一28*28大小
    img = np.array(IM).flatten()  # 拉成一维向量
    return img


# 获取两张图片之间的欧式距离
def get_dis(img1, img2):
    """
    :param img1: 测试数字
    :param img2: 模板数字
    :return: 两个图片间的欧式距离
    """
    return np.sqrt(np.sum(np.square(img1-img2)))


if __name__ == '__main__':
    # 读取10张数字图片作为模板
    tmp_imgs = []
    for i in range(10):
        filename = f'training_set/carNum/%d.bmp' % i      # 车牌数据集的0-9图片作为模板
        img = imread(filename)
        tmp_imgs.append(img)

    # 待匹配样本
    test_imgs = []
    for i in range(10):
        fn = f'training_set/carNum/%d.1.bmp' % i          # 车牌数字0-9的测试集
        test_img = imread(fn)
        test_imgs.append(test_img)

    # 计算待匹配样本与模板样本的距离，取最小值为匹配结果
    count = 0
    for j in range(10):
        j_dis = [get_dis(test_imgs[j], img) for img in tmp_imgs]    # 第j个样本与模板匹配的欧氏距离（下标0）
        j_res = j_dis.index(min(j_dis)) + 1                         # 第j个样本的匹配结果
        print("数字 %d 的匹配结果是%s" % (j+1, j_res))
        if j+1 == j_res:
            count += 1
    print("正确率为:", count/10)
