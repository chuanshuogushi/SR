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

    # 左移和下移进行全插值
    output = np.zeros((left.shape[0]*2 - 1, left.shape[1]*2 - 1))
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            output[2*i][2*j] = (int(left[i][j]) + int(down[i][j])) / 2
            if j != left.shape[1] - 1:
                output[2*i][2*j+1] = (int(left[i][j+1]) + int(down[i][j])) / 2
            if i != left.shape[0] - 1:
                output[2*i+1][2*j] = (int(left[i+1][j]) + int(down[i][j])) / 2
            if j != left.shape[1] - 1 and i != left.shape[0] - 1:
                output[2*i+1][2*j+1] = (int(left[i+1][j+1]) + int(down[i][j])) / 2
    im = Image.fromarray(output)
    im.show()
    im = im.convert('L')
    im.save('l+d_t.png')
    print('done!')

