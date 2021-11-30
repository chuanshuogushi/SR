import cv2
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

wx0 = mpimg.imread('DATA/0/wx_1.bmp')  # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
print(wx0.shape)  # (512, 512, 3)
rgbwx0 = cv2.cvtColor(wx0, cv2.COLOR_BGR2RGB)
plt.imshow(rgbwx0)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.show()

# from PIL import Image
# im = Image.open('DATA/0/wx_0.bmp')
# im.show()
