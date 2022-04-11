"""实现全插值+双线性插值结合"""
#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
from PIL import Image
from utils import total_bilinear_inter

if __name__ == '__main__':
    # 读入左上移和原图的图片
    leftup = cv2.imread('DATA/0/wx_2.bmp')
    leftup = cv2.cvtColor(leftup, cv2.COLOR_BGR2GRAY)  # 化为灰度图

    origin = cv2.imread('DATA/0/wx_0.bmp')
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

    # 左上移和原图进行全插值+双线性插值
    im = total_bilinear_inter(leftup, origin)
    im.show()
    im = im.convert('L')
    im.save('result/o+lu_tb.png')
