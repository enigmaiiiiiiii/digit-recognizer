import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import optim
from sklearn.model_selection import train_test_split

train_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/train.csv').astype('float32')
test_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/test.csv').astype('float32')

# %%
targets_numpy = train_data.label.values
features_numpy = train_data.drop(columns='label').values / 255

features_train, features_test, targets_train, targets_test = train_test_split(
    features_numpy, targets_numpy, test_size=0.2, random_state=42)

batch_size = 100
n_iters = 10000
# num_epochs = n_iters / (len(features_train) / batch_size)

# num_epochs = int(num_epochs)
num_epochs = 3

featuresTrain = torch.from_numpy(features_train).view(-1,1,28,28)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
featuresTest = torch.from_numpy(features_test).view(-1,1,28,28)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
"""将numpy类型转换为tensor"""
# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

data_train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
data_test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)


# %%

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


model = LeNet()
model.train()
lr = 0.01
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4)  # 随机梯度优化
train_loss = []
log_interval = 10
test_losses = []


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
        loss = criterion(outputs, targets)      #有target才有loss
        loss.backward()           #损失反向传播
        optimizer.step()
        train_loss.append(loss.item())
        _, predicted = outputs.max(1)
        """outputs.max(1)返回index,value"""
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

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


def test():
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_test_loader:
            output = model(data)
            """data的size（N，C，H，W），output的size（N，最后一层的维度）"""
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]     #预测值的index
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += target.size(0)
    test_loss /= len(test_data)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data), 100. * correct / total))


if __name__ == '__main__':
    test()
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test()

testTensor = torch.from_numpy(test_data.values / 255).view(-1, 1, 28, 28)
random_index = np.random.randint(0, len(test_data), 16)
predict = model(testTensor[random_index])
predict_result = predict.data.max(1)[1].numpy()

fig, axes = plt.subplots(4, 4, figsize=(8, 8),
                         subplot_kw={'xticks': [], 'yticks': []},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_data.iloc[random_index[i]].values.reshape(28,28), cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(predict_result[i]),
                transform=ax.transAxes, size = 25,color='black')


plt.show()