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


from lrcnn_model import LRCNN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR-path', type=str, required=True, default='DATA/Data_for_SRGAN/HR')
    parser.add_argument('--LR-path', type=str, required=True, default='DATA/Data_for_SRGAN/sub_LR')
    parser.add_argument('--output-dir', type=str, required=True, default='DATA/Data_for_SRGAN/sub_LR_out')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=18374288)
    parser.add_argument('--num-epoch', type=int, default=5)
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

    train_set = Dataset(args.HR_path)   # TODO
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size)

    # TODO eval相关未添加，best_weights等未初始化

    for epoch in range(args.num_epoch):
        model.train()
        epoch_loss = 0 # TODO

