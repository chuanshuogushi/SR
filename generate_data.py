"""由高分辨率生成低分辨率图片，数据集生成
分辨率缩小：2n*2n  ->  n*n
"""
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from utils import reshape_img
from utils import gen_lr
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR-path', type=str, required=True)
    parser.add_argument('--LR-path', type=str, required=True)
    args = parser.parse_args()
    # HR_path = 'DATA\Data_for_SRGAN\HR' # 高分辨率图片文件夹读入路径
    # LR_path = 'DATA\Data_for_SRGAN\sub_LR'  # 低分辨率图片文件夹存放路径

# 批量遍历文件夹中的HR图片
    for hr_name in os.listdir(args.HR_path):
        hr_path = os.path.join(args.HR_path, hr_name)
        hr_name_nosuffix = hr_name.split('.')[0]  # 不要后缀
        lr1_path = os.path.join(args.LR_path, hr_name_nosuffix)
        lr2_path = os.path.join(args.LR_path, hr_name_nosuffix)
        hr = cv2.imread(hr_path)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)  # 化为灰度图
    # step1:将所有HR图片分辨率修改为2m*2n
        hr = reshape_img(hr)
    # step2:得到LR的图片
        lr1, lr2 = gen_lr(hr)
    # step3:图片输出
        lr1 = lr1.convert('L')  # 化为灰度图
        lr2 = lr2.convert('L')
        lr1.save(lr1_path + '-1.png')
        lr2.save(lr2_path + '-2.png')
        # lr1.show()
        # lr2.show()
print('Data Generated.')