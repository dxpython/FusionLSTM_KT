import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ElectricityDataset(Dataset):
    def __init__(self, csv_file, seq_length=24, knowledge_dim=3, feature_cols=None, target_col="OT", transform=None):
        """
        Parameters:
            csv_file: CSV file path
            seq_length: The length of the sliding window, i.e. the number of steps in the input sequence
            knowledge_dim: The dimension of the knowledge vector, by default the statistics of target_col are used
            feature_cols: A list of column names used as features; if None, all columns except "date" and target_col are selected by default
            target_col: The column name where the target value is located
            transform: Optional data preprocessing method
        """
        self.data = pd.read_csv(csv_file)
        if "date" in self.data.columns:
            self.data["date"] = pd.to_datetime(self.data["date"])

        if feature_cols is None:
            feature_cols = [col for col in self.data.columns if col not in ["date", target_col]]
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform

        # 将特征和目标列转换为数值类型
        self.data[feature_cols] = self.data[feature_cols].apply(pd.to_numeric, errors='coerce')
        self.data[target_col] = pd.to_numeric(self.data[target_col], errors='coerce')
        self.data.dropna(subset=feature_cols + [target_col], inplace=True)

        # 提取特征和目标值
        self.features = self.data[feature_cols].values
        self.targets = self.data[target_col].values

        # 对特征和目标采用 z-score 
        self.feature_mean = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std

        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std() + 1e-8
        self.targets = (self.targets - self.target_mean) / self.target_std

        if len(self.features) <= seq_length:
            raise ValueError("Not enough data for the given sequence length.")

        # 计算知识向量
        stats = []
        if knowledge_dim >= 1:
            stats.append(np.mean(self.targets))
        if knowledge_dim >= 2:
            stats.append(np.std(self.targets))
        if knowledge_dim >= 3:
            stats.append(np.max(self.targets))
        if len(stats) < knowledge_dim:
            stats += [0.0] * (knowledge_dim - len(stats))
        self.knowledge = torch.tensor(stats, dtype=torch.float)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, self.knowledge, y


def get_dataloader(csv_file, batch_size=32, seq_length=24, knowledge_dim=3, feature_cols=None, target_col="OT",
                   shuffle=True):
    dataset = ElectricityDataset(csv_file, seq_length, knowledge_dim, feature_cols, target_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    csv_file = "/root/autodl-tmp/fusion/dataset/electricity.csv"
    dataloader = get_dataloader(csv_file, seq_length=24, knowledge_dim=3)
    for batch in dataloader:
        x, knowledge, y = batch
        print("x shape:", x.shape)
        print("knowledge shape:", knowledge.shape)
        print("y shape:", y.shape)
        break
