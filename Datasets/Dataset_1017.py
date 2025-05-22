# Datasets.py

import torch
from torch.utils.data import Dataset


class GeoSpatialDataset(Dataset):
    def __init__(self, dynamic_features, lat, lon, static_features, targets, year,month,seq_len=10):
        """
        自定义地理空间数据集类。

        参数:
            dynamic_features (pd.DataFrame): 动态特征数据。
            lat (pd.Series): 纬度数据。
            lon (pd.Series): 经度数据。
            static_features (pd.DataFrame): 静态特征数据。
            targets (np.array or pd.Series): 目标变量。
            seq_len (int): 序列长度。
        """
        self.dynamic_features = dynamic_features.values  # [num_samples, num_dynamic_features]
        self.lat = lat.values  # [num_samples]
        self.lon = lon.values  # [num_samples]
        self.static_features = static_features.values  # [num_samples, num_static_features]
        self.targets = targets  # [num_samples]
        self.year = year.values
        self.month = month.values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dynamic_features) - self.seq_len

    def __getitem__(self, idx):
        """
        获取第 idx 个样本。

        返回:
            dynamic_x (torch.Tensor): 动态特征序列 [seq_len, num_dynamic_features]
            lat (torch.Tensor): 纬度 [1]
            lon (torch.Tensor): 经度 [1]
            static_features (torch.Tensor): 静态特征 [num_static_features]
            y (torch.Tensor): 目标变量 [1]
        """
        # 动态特征序列
        dynamic_x = self.dynamic_features[idx:idx + self.seq_len]  # [seq_len, num_dynamic_features]

        # 对应时间步的纬度和经度
        lat_sample = self.lat[idx:idx + self.seq_len]  # 标量
        lon_sample = self.lon[idx:idx + self.seq_len]  # 标量
        year_sample = self.year[idx:idx + self.seq_len]
        month_sample = self.month[idx:idx + self.seq_len]

        # 对应时间步的静态特征
        static_sample = self.static_features[idx:idx + self.seq_len]  # [num_static_features]

        # 目标变量
        y = self.targets[idx:idx + self.seq_len]  # 标量  b,seq,1
        # y = y[:,0]

        return (
            torch.tensor(dynamic_x, dtype=torch.float32),  # [seq_len, num_dynamic_features]
            torch.tensor(lat_sample, dtype=torch.float32),  # [1]
            torch.tensor(lon_sample, dtype=torch.float32),  # [1]
            torch.tensor(static_sample, dtype=torch.float32),  # [num_static_features]
            torch.tensor(year_sample, dtype=torch.float32),
            torch.tensor(month_sample, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32) # [1]
            
        )

# ------------------------------非序列任务-----------------------------------------------------
# class GeoSpatialDataset(Dataset):
#     def __init__(self, dynamic_features, lat, lon, static_features, targets,seq_len =1):
#         """
#         自定义地理空间数据集类（非序列任务）。

#         参数:
#             dynamic_features (pd.DataFrame): 动态特征数据。
#             lat (pd.Series): 纬度数据。
#             lon (pd.Series): 经度数据。
#             static_features (pd.DataFrame): 静态特征数据。
#             targets (np.array or pd.Series): 目标变量。
#         """
#         self.dynamic_features = dynamic_features.values  # [num_samples, num_dynamic_features]
#         self.lat = lat.values  # [num_samples]
#         self.lon = lon.values  # [num_samples]
#         self.static_features = static_features.values  # [num_samples, num_static_features]
#         self.targets = targets  # [num_samples]

#     def __len__(self):
#         return len(self.dynamic_features)

#     def __getitem__(self, idx):
#         """
#         获取第 idx 个样本。

#         返回:
#             dynamic_x (torch.Tensor): 动态特征 [1, num_dynamic_features]
#             lat (torch.Tensor): 纬度 [1]
#             lon (torch.Tensor): 经度 [1]
#             static_features (torch.Tensor): 静态特征 [num_static_features]
#             y (torch.Tensor): 目标变量 [1]
#         """
#         # 动态特征（单个时间步）
#         dynamic_x = self.dynamic_features[idx].reshape(1, -1)  # [1, num_dynamic_features]

#         # 对应样本的纬度和经度
#         lat_sample = self.lat[idx]  # 标量
#         lon_sample = self.lon[idx]  # 标量

#         # 对应样本的静态特征
#         static_sample = self.static_features[idx]  # [num_static_features]

#         # 目标变量
#         y = self.targets[idx]  # 标量

#         return (
#             torch.tensor(dynamic_x, dtype=torch.float32),  # [1, num_dynamic_features]
#             torch.tensor(lat_sample, dtype=torch.float32).unsqueeze(0),  # [1]
#             torch.tensor(lon_sample, dtype=torch.float32).unsqueeze(0),  # [1]
#             torch.tensor(static_sample, dtype=torch.float32),  # [num_static_features]
#             torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # [1]
#         )