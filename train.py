"""
lrcnn网络训练
"""
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataset import MyDataset
import torchvision
import numpy as np
import cv2
from PIL import Image
# from utils import AverageMeter
from lrcnn_model import LRCNN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR-path', type=str, default='DATA/T91/sub_LR/1')
    parser.add_argument('--LR-path', type=str, default='DATA/T91/sub_LR/2')
    parser.add_argument('--output-dir', type=str, default='DATA/T91/out')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=18374288)
    parser.add_argument('--num-epoch', type=int, default=5)
    parser.add_argument('--lr', default=0.001)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)  # 设定随机数种子
    model = LRCNN().to(device)
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.Adam([  # 优化器
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr*0.1}
    ], lr=args.lr)

    transforms_imag = torchvision.transforms.ToTensor()
    train_input_root = './DATA/T91/sub_LR/1'
    train_label_root = './DATA/T91/sub_LR/2'
    eval_input_root = './DATA/Set14/sub_LR/1'
    eval_label_root = './DATA/Set14/sub_LR/2'
    dataset_train=MyDataset(train_input_root, train_label_root, transform=transforms_imag)
    trainloader=DataLoader(dataset_train, shuffle=False)

    for epoch in range(args.num_epoch):
        model.train()
        print(epoch)
        print('*'*8)
        # epoch_losses = AverageMeter
        for b_index, (data, label) in enumerate(trainloader):
            x = data.to(device)
            y = label.to(device)
            preds = model(x)
            # vis
            temp = (255*preds).permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)[0]
            cv2.imwrite('./1.jpg', temp)
            loss = criterion(preds, y)
            # epoch_losses.update(loss.item(), len(x))
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()

        # epoch_psnr = AverageMeter()

        # for data in ev







