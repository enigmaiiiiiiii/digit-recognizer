
import torch
from torch import nn
from torch import optim
import math



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
                m.weight.Train_Test_Data.fill_(1)
                m.bias.Train_Test_Data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.Train_Test_Data.fill_(1)
                m.bias.Train_Test_Data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, x):
        x = self.features[:15](x)
        x = x.view(x.size(0), -1)
        x = self.features[15:](x)
        x = torch.sigmoid(x)
        return x

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
