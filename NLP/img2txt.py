# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/3/26 16:04"
__doc__ = """ """

import pytesseract
from PIL import Image

imgs = [r'resource/e.png']
for img in imgs:
    text = pytesseract.image_to_string(Image.open(img), 'chi_sim')
    print(text)
    print('----------------------------------')


