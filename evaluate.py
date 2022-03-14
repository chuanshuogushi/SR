import numpy as np
import math
import cv2
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# def psnr(img1, img2):
#     """两个图片分辨率相同，一个为GT图，一个为待评估图"""
#     mse = np.mean((img1/255. - img2/255.) ** 2)
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# 针对灰度图进行评估
im = cv2.imread('DATA/0/wx_0.bmp')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im1 = cv2.imread('DATA/0/wx_1.bmp')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
p = psnr(im, im1)
s = ssim(im, im1, data_range=255)
