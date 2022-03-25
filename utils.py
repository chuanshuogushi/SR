import cv2
import numpy as np
from PIL import Image

def reshape_img(img):
    '''将图片分辨率修改为2m*2n'''
    if img.shape[0] % 2 == 0:
        if img.shape[1] % 2 == 0:
            pass  # 保持不变
        else:
            img = img[:, :-1]  # 删掉最后一列
    else:
        if img.shape[1] % 2 == 0:
            img = img[:-1, :]  # 删掉最后一行
        else:
            img = img[:-1, :-1]  # 删掉最后一行一列
    return img

def gen_lr(hr):
    """将HR图片下采样,生成2张LR图片,用于亚像素级别数据集生成
    hr.size=(2m,2n)
    lr1.size=lr2.size=(m-1,n-1)"""
    lr1 = np.zeros((int(hr.shape[0] / 2) - 1, int(hr.shape[1] / 2) - 1))  # 初始化低分辨率图像lr1
    lr2 = np.zeros_like(lr1)  # 初始化低分辨率图像lr2
    for i in range(lr1.shape[0]):
        for j in range(lr1.shape[1]):
            lr1[i, j] = (int(hr[2 * i, 2 * j]) + int(hr[2 * i, 2 * j + 1]) + int(hr[2 * i + 1, 2 * j]) + int(hr[2 * i + 1, 2 * j + 1])) / 4
    for i in range(lr2.shape[0]):
        for j in range(lr2.shape[1]):
            lr2[i, j] = (int(hr[2 * i + 1, 2 * j + 1]) + int(hr[2 * i + 1, 2 * j + 2]) + int(hr[2 * i + 2, 2 * j + 1]) + int(hr[2 * i + 2, 2 * j + 2])) / 4
    return lr1, lr2


def total_inter(lr_l, lr_d):
    """全插值法将左移0.5和下移0.5的图片进行插值。也就是左上和右下插值，同超模式采样数据。得到HR。
    lr_left.size=lr_down.size=(m,n)
    hr.size=(2m-1,2n-1)
    """
    hr = np.zeros((lr_l.shape[0] * 2 - 1, lr_l.shape[1] * 2 - 1))
    for i in range(lr_l.shape[0]):
        for j in range(lr_l.shape[1]):
            hr[2 * i][2 * j] = (int(lr_l[i][j]) + int(lr_d[i][j])) / 2
            if j != lr_l.shape[1] - 1:
                hr[2 * i][2 * j + 1] = (int(lr_l[i][j + 1]) + int(lr_d[i][j])) / 2
            if i != lr_l.shape[0] - 1:
                hr[2 * i + 1][2 * j] = (int(lr_l[i + 1][j]) + int(lr_d[i][j])) / 2
            if j != lr_l.shape[1] - 1 and i != lr_l.shape[0] - 1:
                hr[2 * i + 1][2 * j + 1] = (int(lr_l[i + 1][j + 1]) + int(lr_d[i][j])) / 2
    return hr