"""实现全插值+双线性插值结合"""
#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
from PIL import Image
from utils import total_bilinear_inter

if __name__ == '__main__':
    # 读入左移和下移的图片
    left = cv2.imread('DATA/0/wx_3.bmp')
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # 化为灰度图

    down = cv2.imread('DATA/0/wx_2.bmp')
    down = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY)

    # 左移和下移进行全插值+双线性插值
    im = total_bilinear_inter(left, down)
    im.show()
    im = im.convert('L')
    im.save('result/l+d_tb.png')
