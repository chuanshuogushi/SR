"""实现全插值+双线性插值结合"""
#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
from PIL import Image


if __name__ == '__main__':
    # 读入左移和下移的图片
    left = cv2.imread('DATA/0/wx_3.bmp')
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # 化为灰度图

    down = cv2.imread('DATA/0/wx_2.bmp')
    down = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY)

    # 左移和下移进行全插值+双线性插值

    x = np.zeros((left.shape[0], left.shape[1]))
    y = np.zeros((left.shape[0], left.shape[1]))
    # 双线性插值，计算得到2个中间图层
    for i in range(left.shape[0]-1):
        for j in range(left.shape[1]-1):
            x[i][j] = (int(left[i+1, j]) + int(left[i+1, j+1]) + int(down[i, j]) + int(down[i+1, j]))/4
            y[i][j] = (int(left[i, j+1]) + int(left[i+1, j+1]) + int(down[i, j]) + int(down[i, j+1]))/4
    # 显示中间图层
    # x_im = Image.fromarray(x)
    # y_im = Image.fromarray(y)
    # x_im.show()
    # y_im.show()

    # 用中间图层计算最终图层
    l_d_out = np.zeros((left.shape[0] * 2 - 1, left.shape[1] * 2 - 1))
    for i in range(left.shape[0]-1):
        for j in range(left.shape[1]-1):
            if j != 0 and i != 0:
                l_d_out[2*i][2*j] = (int(left[i][j])+int(down[i][j])+int(x[i-1][j])+int(y[i][j-1]))/4
            if j != 0:
                l_d_out[2*i+1][2*j] = (int(left[i+1][j])+int(down[i][j])+int(x[i][j])+int(y[i][j-1]))/4
            if i != 0:
                l_d_out[2*i][2*j+1] = (int(left[i][j+1])+int(down[i][j])+int(x[i-1][j])+int(y[i][j]))/4
            l_d_out[2*i+1][2*j+1] = (int(left[i+1][j+1])+int(down[i][j])+int(x[i][j])+int(y[i][j]))/4
    im = Image.fromarray(l_d_out)
    im.show()
    im = im.convert('L')
    im.save('l+d_tb.png')
