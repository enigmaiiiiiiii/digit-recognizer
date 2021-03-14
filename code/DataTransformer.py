import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def data(batch_size):
    train_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/train.csv').astype('float32')
    test_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/test.csv').astype('float32')

    targets_numpy = train_data.label.values
    features_numpy = train_data.drop(columns='label').values / 255

    features_train, features_test, targets_train, targets_test = train_test_split(
        features_numpy, targets_numpy, test_size=0.2, random_state=42)
    """把数据集分为训练集和测试集"""

    tensor_train_features = torch.from_numpy(features_train).view(-1, 1, 28, 28)
    tensor_train_targets = torch.from_numpy(targets_train).type(torch.LongTensor)
    tensor_test_features = torch.from_numpy(features_test).view(-1, 1, 28, 28)
    tensor_test_targets = torch.from_numpy(targets_test).type(torch.LongTensor)
    predict_tensor = torch.from_numpy(test_data.values / 255).view(-1, 1, 28, 28)

    """将numpy类型转换为tenso
    r"""
    # Pytorch train and test sets

    train = torch.utils.data.TensorDataset(tensor_train_features, tensor_train_targets)
    test = torch.utils.data.TensorDataset(tensor_test_features, tensor_test_targets)

    data_train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    data_test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return data_train_loader, data_test_loader
