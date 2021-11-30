import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
from PIL import Image
# def total_interpolation(input, output, direction, distance):
#     """input: ndarray格式
#     output
#     direction
#     distance
#     """


if __name__ == '__main__':
    origin = cv2.imread('DATA/0/wx_0.bmp')
    right = cv2.imread('DATA/0/wx_1.bmp')  # 读入后是np.array

    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)  # 化为灰度图
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    (mean0, stddv0) = cv2.meanStdDev(origin)  # 计算均值和方差
    (mean1, stddv1) = cv2.meanStdDev(right)

    # 右移0.5图和原图进行全插值
    output = np.zeros((origin.shape[0] * 2, origin.shape[1] * 2))  # 初始化大图
    for i in range(origin.shape[0]):  # 遍历大图和小图所有像素
        for j in range(origin.shape[1]):
            if j == (origin.shape[1] - 1):  # 最后一列不要了，为构成2m×2n像素
                continue
            else:
                if j != 0:  # 防止数组索引小于0
                    """必须加int，否则会发生溢出"""
                    output[2 * i][2 * j] = (int(origin[i][j]) + int(right[i][j - 1])) / 2
                    output[2 * i + 1][2 * j] = (int(origin[i][j]) + int(right[i][j - 1])) / 2
                output[2*i][2*j+1] = (int(origin[i][j]) + int(right[i][j])) / 2
                output[2*i+1][2*j+1] = (int(origin[i][j]) + int(right[i][j])) / 2
    for i in range(origin.shape[0]):  # 单独考虑大图第一列
        output[2*i][0] = origin[i][0]  # 直接等于小图第一列
        output[2*i+1][0] = origin[i][0]
    im = Image.fromarray(output)
    im.show()
    im = im.convert('L')
    im.save('o+r_t.png')
    print('done!')
    new = cv2.imread('new.png')
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    (mean2, stddv2) = cv2.meanStdDev(new)
    print(mean0,stddv0)
    print(mean1,stddv1)
    print(mean2,stddv2)
    # print(wx0.shape)  # (478,638)
    # rgbwx0 = cv2.cvtColor(wx0, cv2.COLOR_BGR2RGB)  # 转换通道，显示图片
    # plt.imshow(rgbwx0)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
