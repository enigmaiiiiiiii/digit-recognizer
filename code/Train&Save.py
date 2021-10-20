import torch
from torch import nn
from torch import optim
import localmodel
from DataTransformer import data_factory
from torch.utils.tensorboard import SummaryWriter
import os

batch_size = 100
train_data_path = r'./digit-recognizer/train.csv'
test_data_path = r'./digit-recognizer/test.csv'
digit_data = data_factory(train_data_path, test_data_path)

data_train_loader, data_test_loader= digit_data.train_test_dataloader(batch_size)

train_size = len(data_train_loader.dataset)
if data_test_loader is not None:
    test_size = len(data_test_loader.dataset)
writer = SummaryWriter(os.getcwd() + '\\log1')

n_iters = 10000
num_epochs = n_iters / (train_size / batch_size)
num_epochs = int(num_epochs)
lr = 0.003
print_step = 10
model = localmodel.Net()
model.train()
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=lr)


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
        """分割线，以上为模型完成了一次循环，
        以下为训练信息输出"""
        _, predicted = outputs.max(1)
        """outputs.max(1)返回index,value"""
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = correct / total
        writer.add_scalar('acc_train', train_acc)

        if batch_idx % print_step == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(inputs),
                    train_size,
                    100. * batch_idx / len(data_train_loader),
                    loss.item()))
            print(batch_idx,
                  train_size,
                  'Acc: {0:4f}%({1}/{2})'.format(100. * correct / total,
                                                 correct,
                                                 total))
    writer.add_scalar('acc_train', train_acc, epoch)


def test(epoch):
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
    test_loss /= test_size
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, 100. * correct / total))
    writer.add_scalar('acc_test', test_acc, epoch)


for epoch in range(1, num_epochs + 1):
    train(epoch)
    # test(epoch)

if os.path.isdir('.\\TrainedModels') is False:
    os.makedirs('.\\TrainedModels')
torch.save(model, '.\\TrainedModels\\TrainWithAllData.pth')











