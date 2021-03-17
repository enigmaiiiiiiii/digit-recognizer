# %%

"""module和数据导入"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import math
import random
from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
from collections import OrderedDict


batch_size = 100
n_iters = 10000
# num_epochs = n_iters / (len(features_train) / batch_size)
# num_epochs = int(num_epochs)
num_epochs = 5

train_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/train.csv').astype('float32')
test_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/test.csv').astype('float32')

targets_numpy = train_data.label.values
features_numpy = train_data.drop(columns='label').values / 255
testTensor = torch.from_numpy(test_data.values / 255).view(-1, 1, 28, 28)

features_train, features_test, targets_train, targets_test = train_test_split(
    features_numpy, targets_numpy, test_size=0.2, random_state=42)

featuresTrain = torch.from_numpy(features_train).view(-1, 1, 28, 28)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test).view(-1, 1, 28, 28)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
"""将numpy类型转换为tensor"""
# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

data_train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
data_test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)




class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(16 * 4 * 4, 10)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc3(x))
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Dropout(p=0.4),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.4),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.4),
            nn.Linear(128, 10),
        )
        # self.norm1 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(p=0.25)
        # self.flatten2 = nn.Linear(128, 10)

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, x):
        x = self.features[:15](x)
        x = x.view(x.size(0), -1)
        x = self.features[15:](x)
        x = torch.sigmoid(x)
        return x


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # convolution 1
        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # maxpool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # dropout 1
        self.dropout1 = nn.Dropout(0.25)

        # convolution 2
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # maxpool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # dropout 2
        self.dropout2 = nn.Dropout(0.25)

        # linear 1
        self.fc1 = nn.Linear(32 * 5 * 5, 256)

        # dropout 3
        self.dropout3 = nn.Dropout(0.25)

        # linear 2
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.c1(x)  # [BATCH_SIZE, 16, 24, 24]
        out = self.relu1(out)
        out = self.maxpool1(out)  # [BATCH_SIZE, 16, 12, 12]
        out = self.dropout1(out)

        out = self.c2(out)  # [BATCH_SIZE, 32, 10, 10]
        out = self.relu2(out)
        out = self.maxpool2(out)  # [BATCH_SIZE, 32, 5, 5]
        out = self.dropout2(out)

        out = out.view(out.size(0), -1)  # [BATCH_SIZE, 32*5*5=800]
        out = self.fc1(out)  # [BATCH_SIZE, 256]
        out = self.dropout3(out)
        out = self.fc2(out)  # [BATCH_SIZE, 10]

        return out


model = Net()
model.train()
lr = 0.03
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=lr)  # 随机梯度优化
train_loss = []
log_interval = 10
test_losses = []
writer = SummaryWriter(os.getcwd() + '\\log1')


def train(epoch):
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_train_loader):

        """
        batch_idx批次编号
        inputs 批处理数据集size等于dataloader中的batch_size
        targets目标集
        """
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # 有target才有loss
        loss.backward()  # 损失反向传播
        optimizer.step()
        train_loss.append(loss.item())
        _, predicted = outputs.max(1)
        """outputs.max(1)返回index,value"""
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = correct / total
        writer.add_scalar('acc_train', train_acc)

        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_data),
                    100. * batch_idx / len(data_train_loader),
                    loss.item()))
            print(batch_idx,
                  len(train_data),
                  'Acc: {0:4f}%({1}/{2})'.format(100. * correct / total,
                                                 correct,
                                                 total))
    writer.add_scalar('acc_train', train_acc, epoch)


def test():
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_test_loader:
            output = model(data)
            """data的size（N，C，H，W），output的size（N，最后一层的维度）"""
            test_loss += criterion(output, target).item()
            pred = output.Train_Test_Data.max(1, keepdim=True)[1]
            correct += pred.eq(target.Train_Test_Data.view_as(pred)).sum()
            total += target.size(0)  # 数据类型int
            test_acc = torch.true_divide(correct, total)
    test_loss /= len(test_data)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, 100. * correct / total))
    writer.add_scalar('acc_test', test_acc, epoch)


for epoch in range(1, num_epochs + 1):
    train(epoch)
    test()
