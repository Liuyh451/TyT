import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from torch.utils.data import Dataset, random_split


class Reanalysis3DCNN(nn.Module):
    def __init__(self, in_channels=9, input_size=40):
        super(Reanalysis3DCNN, self).__init__()

        # 时空特征提取模块
        self.conv_block = nn.Sequential(
            # 输入形状: (batch*5, 9, 40, 40)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (batch*5, 32, 20, 20)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (batch*5, 64, 10, 10)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> (batch*5, 128, 5, 5)
        )

        # 全连接层保持时间步独立性
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        """
        :param x: 输入张量形状 (batch_size, timesteps=5, channels=9, height=40, width=40)
        :return: 输出张量形状 (batch_size, timesteps=5, 128)
        """
        # 保存原始形状
        batch_size, timesteps, C, H, W = x.size()
        # 合并批次和时间维度
        x = x.view(batch_size * timesteps, C, H, W)  # (batch*5, 9, 40, 40)
        # 卷积特征提取
        x = self.conv_block(x)  # -> (batch*5, 128, 5, 5)
        # 展平特征
        x = x.view(x.size(0), -1)  # (batch*5, 128*5*5)
        # 全连接处理
        x = self.fc_block(x)  # -> (batch*5, 128)
        # 恢复时间维度
        x = x.view(batch_size, timesteps, -1)  # (batch, 5, 128)
        return x


class PairedTyphoonDataset(Dataset):
    def __init__(self, era5_data, bst_data, seq_len=5, transform='minmax'):
        """
        era5_data: ndarray of shape (T, 9, 40, 40)
        bst_data: ndarray of shape (T, 4)
        """
        assert era5_data.shape[0] == bst_data.shape[0]
        self.seq_len = seq_len

        # 归一化 ERA5 数据
        self.era5 = self.normalize(era5_data, method=transform)
        scaler = MinMaxScaler()
        self.bst = scaler.fit_transform(bst_data)  # 每列归一化
        # self.bst = self.normalize(bst_data)  # 一般不归一化 label，除非数值范围差异极大

        # 构造样本列表
        self.samples = []
        for i in range(len(self.era5) - seq_len+1):
            x_seq = self.era5[i:i + seq_len]
            y_seq = self.bst[i:i + seq_len]  # 输入序列
            y_target = self.bst[i + seq_len - 1]  # 用最后一个时间步作为预测目标
            self.samples.append((x_seq, y_seq, y_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, z = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(z,
                                                                                                        dtype=torch.float32)

    def normalize(self, data, method='minmax'):
        if method == 'minmax':
            min_val = data.min(axis=(0, 2, 3), keepdims=True)
            max_val = data.max(axis=(0, 2, 3), keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'zscore':
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
            std = data.std(axis=(0, 2, 3), keepdims=True)
            return (data - mean) / (std + 1e-8)
        else:
            raise ValueError("Unsupported normalization method.")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 0,2,4,...
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 1,3,5,...
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerEncoderModule(nn.Module):
    def __init__(self, input_dim=132, seq_len=5, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, norm_first=True  # LayerNorm在前
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(input_dim * seq_len, 4)  # 输出轨迹预测 (lat, lon, pressure, wind)

    def forward(self, x):
        # x: (batch_size, seq_len, 132)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, 132)
        x = self.flatten(x)  # (batch_size, seq_len * 132)
        out = self.fc_out(x)  # (batch_size, 4)
        return out


class TytModel(nn.Module):
    """
    台风轨迹预测整体模型

    参数：
        cnn_in_channels: int, CNN输入通道数（默认气象要素数量）
        cnn_input_size: int, CNN输入空间维度
        seq_len: int, 时序特征序列长度
        combine_feature_dim: int, 需要拼接的辅助特征维度
    """

    def __init__(self, cnn_in_channels=9, cnn_input_size=40, seq_len=5, combine_feature_dim=4):
        super().__init__()
        # 气象数据特征提取模块
        self.cnn = Reanalysis3DCNN(in_channels=cnn_in_channels, input_size=cnn_input_size)

        # 轨迹预测Transformer模块
        self.transformer = TransformerEncoderModule(
            input_dim=128 + combine_feature_dim,  # CNN特征 + 辅助特征
            seq_len=seq_len
        )

    def forward(self, era_data, track_features):
        """
        前向传播过程

        参数：
            era_data: torch.Tensor, ERA再分析数据
                形状 (batch_size, seq_len,9, 40, 40)
            combine_features: torch.Tensor, 辅助特征数据
                形状 (batch_size, seq_len, combine_feature_dim)

        返回：
            torch.Tensor, 预测结果形状 (batch_size, 4)
        """
        # 提取气象空间特征
        cnn_features = self.cnn(era_data)  # (batch_size,5, 128)
        # 特征拼接
        combined = torch.cat([cnn_features, track_features], dim=-1)  # (batch_size, seq_len, 132)
        # 轨迹预测
        return self.transformer(combined)
class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='min'):
        """
        patience: 允许的连续不改善 epoch 数
        delta:    认为指标显著变化的最小阈值
        mode:     'min' 监控损失下降，'max' 监控准确率上升
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score > self.best_score - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        elif self.mode == 'max':
            if current_score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.counter = 0
        return self.early_stop


def plot_loss_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像到文件
    plt.savefig('loss_curves.png')
    plt.show()