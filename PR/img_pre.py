# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/2/24 21:49"
__doc__ = """ 图像预处理"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from scipy import misc



imgPath = r'D:\Pyproject\OIP.jpg'
# 1.PIL
img1 = Image.open(imgPath).convert('L')      # 不直接返回ndarray对象
img1 = np.array(img1)           # np.array()
print(type(img1))
print(img1.shape)

# 2.matplotlib
img2 = mpimg.imread(imgPath)    # ndarray, 通道顺序: RGB
print(type(img2))
print(img2.shape)

# 3.skimage
img3 = io.imread(imgPath)       # ndarray, 通道顺序: RGB
print(type(img3))
print(img3.shape)

# 4.misc.imread
img4 = io.imread(imgPath)       # ndarray, 通道顺序: RGB
print(type(img4))
print(img4.shape)

# 5.opencv
# img5 = cv2.imread(imgPath)    # ndarray, 通道顺序: BGR
# print(type(img5))
# print(img5.shape)

plt.subplot(231)
plt.imshow(img1)
plt.subplot(232)
plt.imshow(img2)
plt.subplot(233)
plt.imshow(img3)
plt.subplot(234)
plt.imshow(img4)
plt.subplot(235)
plt.imshow(img5)
plt.show()

