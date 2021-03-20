import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class data_factory:

    def __init__(self, train_data_path=None, predict_data_path=None):

        self.train_data_path = train_data_path
        self.predict_data_path = predict_data_path
        self.train_data = pd.read_csv(self.train_data_path).astype('float32')
        self.test_data = pd.read_csv(self.predict_data_path).astype('float32')

    def train_test_dataloader(self, batch_size, test_size=False):
        """为了训练和测试，创建pytorh模型用的DataLoader"""

        targets_numpy = self.train_data.label.values
        features_numpy = self.train_data.drop(columns='label').values / 255

        if not test_size:
            tensor_train_features = torch.from_numpy(features_numpy).view(-1, 1, 28, 28)
            tensor_train_targets = torch.from_numpy(targets_numpy).type(torch.LongTensor)

            train = TensorDataset(tensor_train_features, tensor_train_targets)
            data_train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
            data_test_loader = None



        else:
            features_train, features_test, targets_train, targets_test = train_test_split(
                features_numpy, targets_numpy, test_size=test_size, random_state=42)
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

    def predict_dataloader(self, batch_size):
        """为了预测结果，创建pytorh模型用的DataLoader"""
        predict_values = pd.read_csv(self.predict_data_path).astype('float32')
        predict_tensor = torch.from_numpy(predict_values.values / 255).view(-1, 1, 28, 28)
        predict_dataset = TensorDataset(predict_tensor)
        predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

        return predict_loader



