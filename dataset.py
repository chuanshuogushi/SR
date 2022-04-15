from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import torch
import torchvision
import os
from PIL import Image
from matplotlib import pyplot as plt

# 对读取的图片采取的处理方法，详情自行搜索transforms的用法
# transforms_imag = torchvision.transforms.Compose([torchvision.transforms.Resize([64, 64]),
#                                                   torchvision.transforms.ToTensor()])
# transforms_imag = torchvision.transforms.ToTensor()
# # 输入与标签图片所在的目录
# input_root = './DATA/T91/sub_LR/1'
# label_root = './DATA/T91/sub_LR/2'


class MyDataset(Dataset):  # 继承了Dataset子类
    def __init__(self, input_root, label_root, transform=None):
        # 分别读取输入/标签图片的路径信息
        self.input_root = input_root
        self.input_files = os.listdir(input_root)  # 列出指定路径下的所有文件

        self.label_root = label_root
        self.label_files = os.listdir(label_root)

        self.transforms = transform

    def __len__(self):
        # 获取数据集大小
        return len(self.input_files)

    def __getitem__(self, index):
        # 根据索引(id)读取对应的图片
        input_img_path = os.path.join(self.input_root, self.input_files[index])
        print(input_img_path)
        input_img = Image.open(input_img_path)
        # 视频教程使用skimage来读取的图片，但我在之后使用transforms处理图片时会报错
        # 所以在此我修改为使用PIL形式读取的图片

        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = Image.open(label_img_path)

        if self.transforms:
            # transforms方法如果有就先处理，然后再返回最后结果
            input_img = self.transforms(input_img)
            label_img = self.transforms(label_img)

        return (input_img, label_img)  # 返回成对的数据

        ###把以下代码放在return前，反注释后运行
        # # test only for PIL#
        # input_img.show()
        # label_img.show()
        # # test only for PIL#

