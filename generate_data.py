"""由高分辨率生成低分辨率图片，数据集生成
分辨率缩小：2n*2n  ->  n*n
"""
import cv2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    # HR_path = 'DATA/0/wx_0.bmp'
    # LR_path = 'DATA'

# 读入图片
    hr = cv2.imread('DATA/0/wx_0.bmp')
    # right = cv2.imread('DATA/0/wx_1.bmp')  # 读入后是np.array
    # left = cv2.imread('DATA/0/wx_3.bmp')
    # down = cv2.imread('DATA/0/wx_2.bmp')

    # down = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY)
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)  # 化为灰度图
    # right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # 化为灰度图
    hr_shape = hr.shape


# step1:将所有HR图片分辨率修改为2m*2n
    if hr.shape[0] % 2 == 0:
        if hr.shape[1]%2 == 0:
            pass  # 保持不变
        else:
            hr = hr[:, :-1]  # 删掉最后一列
    else:
        if hr.shape[1] % 2 == 0:
            hr = hr[:-1, :]  # 删掉最后一行
        else:
            hr = hr[:-1, :-1]  # 删掉最后一行一列
# step2:得到LR_1的图片
    lr = np.zeros((int(hr.shape[0] / 2), int(hr.shape[1] / 2)))  # 初始化低分辨率图像
    for i in range(int(hr.shape[0] / 2)):
        for j in range(int(hr.shape[1] / 2)):
            lr[i, j] = (int(hr[2*i, 2*j]) + int(hr[2*i, 2*j+1]) + int(hr[2*i+1, 2*j]) + int(hr[2*i+1, 2*j+1])) / 4

