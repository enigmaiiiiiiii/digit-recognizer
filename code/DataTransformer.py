import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def Train_Test_Data(batch_size):
    train_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/train.csv').astype('float32')

    targets_numpy = train_data.label.values
    features_numpy = train_data.drop(columns='label').values / 255

    features_train, features_test, targets_train, targets_test = train_test_split(
        features_numpy, targets_numpy, test_size=0.2, random_state=42)
    """把数据集分为训练集和测试集"""

    tensor_train_features = torch.from_numpy(features_train).view(-1, 1, 28, 28)
    tensor_train_targets = torch.from_numpy(targets_train).type(torch.LongTensor)
    tensor_test_features = torch.from_numpy(features_test).view(-1, 1, 28, 28)
    tensor_test_targets = torch.from_numpy(targets_test).type(torch.LongTensor)

    """将numpy类型转换为tensor"""
    # Pytorch train and test sets

    train = TensorDataset(tensor_train_features, tensor_train_targets)
    test = TensorDataset(tensor_test_features, tensor_test_targets)

    data_train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    data_test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return data_train_loader, data_test_loader


def predict_data(batch_size):
    predict_data = pd.read_csv(r'D:/JupyterProject/digit-recognizer/test.csv').astype('float32')
    predict_tensor = torch.from_numpy(predict_data.values / 255).view(-1, 1, 28, 28)
    predict_dataset = TensorDataset(predict_tensor)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    return predict_loader
