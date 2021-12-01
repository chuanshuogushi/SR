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
    left = cv2.imread('DATA/0/wx_3.bmp')
    down = cv2.imread('DATA/0/wx_2.bmp')

    down = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY)
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)  # 化为灰度图
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)  # 化为灰度图

    (mean0, stddv0) = cv2.meanStdDev(origin)  # 计算均值和方差
    (mean1, stddv1) = cv2.meanStdDev(right)

    # 右移0.5图和原图进行全插值
    o_r_out = np.zeros((origin.shape[0] * 2, origin.shape[1] * 2))  # 初始化大图,origin_right_output
    for i in range(origin.shape[0]):  # 遍历大图和小图所有像素
        for j in range(origin.shape[1]):
            if j == (origin.shape[1] - 1):  # 最后一列不要了，为构成2m×2n像素
                continue
            else:
                if j != 0:  # 防止数组索引小于0
                    """必须加int，否则会发生溢出"""
                    o_r_out[2 * i][2 * j] = (int(origin[i][j]) + int(right[i][j - 1])) / 2
                    o_r_out[2 * i + 1][2 * j] = (int(origin[i][j]) + int(right[i][j - 1])) / 2
                o_r_out[2 * i][2 * j + 1] = (int(origin[i][j]) + int(right[i][j])) / 2
                o_r_out[2 * i + 1][2 * j + 1] = (int(origin[i][j]) + int(right[i][j])) / 2
    for i in range(origin.shape[0]):  # 单独考虑大图第一列
        o_r_out[2 * i][0] = origin[i][0]  # 直接等于小图第一列
        o_r_out[2 * i + 1][0] = origin[i][0]
    im = Image.fromarray(o_r_out)
    im.show()
    im = im.convert('L')  # 转换后才能保存
    im.save('o+r_t.png')
    # 读入输出的图片，化成灰度图计算方差和均值
    new = cv2.imread('o+r_t.png')
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    (mean2, stddv2) = cv2.meanStdDev(new)
    print(mean0, stddv0)
    print(mean1, stddv1)
    print(mean2, stddv2)

    # 将left和down进行全插值
    l_d_out = np.zeros((left.shape[0] * 2 - 1, left.shape[1] * 2 - 1))
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            l_d_out[2 * i][2 * j] = (int(left[i][j]) + int(down[i][j])) / 2
            if j != left.shape[1] - 1:
                l_d_out[2 * i][2 * j + 1] = (int(left[i][j + 1]) + int(down[i][j])) / 2
            if i != left.shape[0] - 1:
                l_d_out[2 * i + 1][2 * j] = (int(left[i + 1][j]) + int(down[i][j])) / 2
            if j != left.shape[1] - 1 and i != left.shape[0] - 1:
                l_d_out[2 * i + 1][2 * j + 1] = (int(left[i + 1][j + 1]) + int(down[i][j])) / 2
    im = Image.fromarray(l_d_out)
    im.show()
    im = im.convert('L')
    im.save('l+d_t.png')
    print('done!')

    # print(wx0.shape)  # (478,638)
    # rgbwx0 = cv2.cvtColor(wx0, cv2.COLOR_BGR2RGB)  # 转换通道，显示图片
    # plt.imshow(rgbwx0)  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
