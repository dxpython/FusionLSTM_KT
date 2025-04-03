import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class WeatherDataset(Dataset):
    def __init__(self, csv_file, seq_length=24, knowledge_dim=3, feature_cols=None, target_col="OT", transform=None):
        """
        参数：
          csv_file: CSV 文件路径
          seq_length: 滑动窗口的长度，即输入序列的步数
          knowledge_dim: 知识向量的维度，默认使用 target_col 的统计信息（均值、标准差、最大值）
          feature_cols: 用作特征的列名列表；若为 None，则默认选取除 "date" 与 target_col 外的所有列
          target_col: 目标值所在的列名（预测该列的未来值）
          transform: 可选的数据预处理方法
        """
        self.data = pd.read_csv(csv_file)
        
        # 如果存在日期列，将其转换为 datetime 类型
        if "date" in self.data.columns:
            self.data["date"] = pd.to_datetime(self.data["date"])
        
        # 如果未指定特征列，则选取除 "date" 和目标列之外的所有列
        if feature_cols is None:
            feature_cols = [col for col in self.data.columns if col not in ["date", target_col]]
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform
        
        # 将特征和目标列转换为数值类型（无法转换的值置为 NaN），再删除含 NaN 的行
        self.data[feature_cols] = self.data[feature_cols].apply(pd.to_numeric, errors='coerce')
        self.data[target_col] = pd.to_numeric(self.data[target_col], errors='coerce')
        self.data.dropna(subset=feature_cols + [target_col], inplace=True)
        
        # 提取特征和目标值
        self.features = self.data[feature_cols].values  # shape: [n_samples, num_features]
        self.targets = self.data[target_col].values       # shape: [n_samples]
        
        # 检查是否有足够数据
        if len(self.features) <= seq_length:
            raise ValueError("Not enough data for the given sequence length.")
        
        # 对特征进行 min-max 归一化
        self.feature_min = self.features.min(axis=0)
        self.feature_max = self.features.max(axis=0)
        self.features = (self.features - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)
        
        # 对目标进行 min-max 归一化
        self.target_min = self.targets.min()
        self.target_max = self.targets.max()
        self.targets = (self.targets - self.target_min) / (self.target_max - self.target_min + 1e-8)
        
        # 计算知识向量（使用归一化后的目标值的统计信息：均值、标准差、最大值）
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
        # 每个样本为连续 seq_length 长度的时间序列，目标为后续一个时刻的目标值
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # 获取输入特征序列和对应目标值
        x = self.features[idx: idx + self.seq_length]  # shape: [seq_length, num_features]
        y = self.targets[idx + self.seq_length]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, self.knowledge, y

def get_dataloader(csv_file, batch_size=32, seq_length=24, knowledge_dim=3, feature_cols=None, target_col="OT", shuffle=True):
    dataset = WeatherDataset(csv_file, seq_length, knowledge_dim, feature_cols, target_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    csv_file = "/root/autodl-tmp/fusion/dataset/weather.csv"
    dataloader = get_dataloader(csv_file, seq_length=24, knowledge_dim=3)
    for batch in dataloader:
        x, knowledge, y = batch
        print("x shape:", x.shape)             
        print("knowledge shape:", knowledge.shape)  
        print("y shape:", y.shape)
        break
