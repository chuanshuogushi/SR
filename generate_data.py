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
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)  # 化为灰度图
# step1:将所有HR图片分辨率修改为2m*2n
    if hr.shape[0] % 2 == 0:
        if hr.shape[1] % 2 == 0:
            pass  # 保持不变
        else:
            hr = hr[:, :-1]  # 删掉最后一列
    else:
        if hr.shape[1] % 2 == 0:
            hr = hr[:-1, :]  # 删掉最后一行
        else:
            hr = hr[:-1, :-1]  # 删掉最后一行一列
# step2:得到LR的图片
    lr1 = np.zeros((int(hr.shape[0] / 2)-1, int(hr.shape[1] / 2)-1))  # 初始化低分辨率图像lr1
    lr2 = np.zeros_like(lr1)  # 初始化低分辨率图像lr2
    for i in range(lr1.shape[0]):
        for j in range(lr1.shape[1]):
            lr1[i, j] = (int(hr[2*i, 2*j]) + int(hr[2*i, 2*j+1]) + int(hr[2*i+1, 2*j]) + int(hr[2*i+1, 2*j+1])) / 4
    for i in range(lr2.shape[0]):
        for j in range(lr2.shape[1]):
            lr2[i, j] = (int(hr[2*i+1, 2*j+1])+int(hr[2*i+1, 2*j+2])+int(hr[2*i+2, 2*j+1])+int(hr[2*i+2, 2*j+2])) / 4
# step3:图片输出
    lr1_img = Image.fromarray(lr1)
    lr2_img = Image.fromarray(lr2)
    lr1_img = lr1_img.convert('L')  # 化为灰度图
    lr2_img = lr2_img.convert('L')
    lr1_img.save('DATA/LR_imgs/lr1.png')
    lr2_img.save('DATA/LR_imgs/lr2.png')
    lr1_img.show()
    lr2_img.show()
