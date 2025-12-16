# 基于clean_ppg_dataset.npy文件，训练模型
from function_lib import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
import math



class BPDataset(Dataset):
    def __init__(self, data_list, is_train=True):
        self.data_list = data_list
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # --- 1. 处理时序信号 (双通道) ---
        # 假设信号长度已经对齐（例如都是 1000 点）
        p_dist = item['ppg_distal'].astype(np.float32)
        p_prox = item['ppg_proximal'].astype(np.float32)
        
        # 简单的信号归一化 (Min-Max 到 0~1)
        # 注意：要在每个通道分别做，防止幅度差异丢失
        p_dist = (p_dist - p_dist.min()) / (p_dist.max() - p_dist.min() + 1e-6)
        p_prox = (p_prox - p_prox.min()) / (p_prox.max() - p_prox.min() + 1e-6)
        # p_prox = p_prox - p_dist
        
        # 堆叠成 (2, Length)
        # PyTorch Conv1d 输入格式: (Batch, Channel, Length)
        signals = np.stack([p_dist, p_prox], axis=0) 
        if self.is_train:
            # 1. 随机幅度缩放 (Scale)
            scale = np.random.uniform(0.9, 1.1)
            signals = signals * scale
            
            # 2. 随机高斯噪声 (Noise)
            noise = np.random.normal(0, 0.01, signals.shape) # 0.01 视归一化情况而定
            signals = signals + noise
            
            # 3. 随机时间平移 (Shift) - 模拟截取时的偏差
            shift = np.random.randint(-10, 10)
            signals = np.roll(signals, shift, axis=1)
        x_signal = torch.tensor(signals, dtype=torch.float32)
        
        # --- 2. 处理标量特征 (Aux Features) ---
        # 我们把所有辅助信息拼成一个向量：[HR, gender, activity, height, weight, age]
        # 注意：神经网络对数值范围敏感，必须归一化！
        gender_mapping = {'male': 1, 'female': 0, 'Male': 1, 'Female': 0}

        hr = float(item['HR']) / 200.0         # 假设最大心率200
        gen = float(gender_mapping[item['gender']])            # 0或1
        act = float(item['activity']) / 3.0    # 假设活动是 0,1,2
        h = float(item['height']) / 200.0      # 归一化身高
        w = float(item['weight']) / 150.0      # 归一化体重
        age = float(item['age']) / 100.0       # 归一化年龄
        
        features = np.array([hr, gen, act, h, w, age], dtype=np.float32)
        x_feat = torch.tensor(features, dtype=torch.float32)
        
        # --- 3. 处理标签 (Label) ---
        # 预测目标：SBP 和 DBP
        sbp = float(item['SBP'])
        dbp = float(item['DBP'])
        
        # 标签通常不需要归一化，或者归一化后输出时再反归一化
        # 这里直接预测真实值
        y_target = torch.tensor([sbp, dbp], dtype=torch.float32)
        
        return x_signal, x_feat, y_target

# --- 1. 定义残差块 (Basic Block) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 捷径 (Shortcut/Skip Connection)
        # 如果维度变了 (stride>1 或 in!=out)，捷径也要通过 1x1 卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 核心：输入 x 直接加到输出上
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- 2. 主网络 ResNet ---
class ResNetPPG(nn.Module):
    def __init__(self):
        super(ResNetPPG, self).__init__()
        
        # Stem 层: 快速降维，处理 7500 的超长输入
        # 7500 -> 1875 (Stride=4)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # 1875 -> 938
        )
        
        # 残差层堆叠 (ResNet-18 风格)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)  # 长度减半 938->469
        self.layer3 = self._make_layer(128, 256, stride=2) # 长度减半 469->235
        self.layer4 = self._make_layer(256, 512, stride=2) # 长度减半 235->118
        
        # 全局平均池化 (GAP): 把 [B, 512, 118] 变成 [B, 512]
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 辅助特征分支
        self.aux_branch = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU()
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def _make_layer(self, in_c, out_c, stride):
        # 一个 Layer 包含 2 个 Residual Block
        return nn.Sequential(
            ResidualBlock(in_c, out_c, stride),
            ResidualBlock(out_c, out_c, stride=1)
        )

    def forward(self, x_sig, x_feat):
        # x_sig: [B, 2, 7500]
        
        x = self.stem(x_sig)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # -> [B, 512, 118]
        
        x = self.gap(x).flatten(1) # -> [B, 512]
        
        # 融合
        aux = self.aux_branch(x_feat)
        combined = torch.cat([x, aux], dim=1)
        
        return self.regressor(combined)

# ==========================================================================================
def main():
    # 配置日志
    logging.info(f'==========================================================================================')
    logging.info(f'Running training_on_clean_data.py')
    logging.info(f'==========================================================================================')

    logging.info(f'using CNN + MLP model, 1 PPG channel + 1 differential PPG')


    # 1. 加载所有数据
    all_samples = np.load('clean_ppg_dataset.npy', allow_pickle=True)

    # 2. 提取所有 subject_id 以便进行划分
    # 注意：你的 subject_id 可能是字符串或数字，这里确保统一
    subject_ids = list(set([str(sample['subject_id']) for sample in all_samples]))
    subject_ids.sort()

    # 3. 按人头划分 (80% 训练, 20% 验证)
    np.random.seed(42)
    np.random.shuffle(subject_ids)

    split_idx = int(len(subject_ids) * 0.8)
    train_ids = subject_ids[:]
    val_ids = subject_ids[:]

    logging.info(f"Total subjects: {len(subject_ids)}, Training subjects: {len(train_ids)}, Validation subjects: {len(val_ids)}")

    # 4. 分拣数据
    train_data = [s for s in all_samples if str(s['subject_id']) in train_ids]
    val_data = [s for s in all_samples if str(s['subject_id']) in val_ids]

    # ==========================================================================================
    # --- 初始化 ---
    logging.info(f'==========================================================================================')
    logging.info(f'Training on {len(train_data)} samples, validating on {len(val_data)} samples')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 数据集与Loader
    train_dataset = BPDataset(train_data, is_train=False)
    val_dataset = BPDataset(val_data, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型、损失、优化器
    model = ResNetPPG().to(device)
    criterion = nn.MSELoss() # 回归任务常用均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # --- 开始训练 ---
    epochs = 50
    best_val_loss = float('inf')
    patience = 5  # 容忍多少轮不下降
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for x_sig, x_feat, y_true in train_loader:
            # 搬运到 GPU
            x_sig = x_sig.to(device)
            x_feat = x_feat.to(device)
            y_true = y_true.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            y_pred = model(x_sig, x_feat)
            
            # 计算 Loss
            loss = criterion(y_pred, y_true)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # --- 验证环节 ---
        model.eval()
        val_loss = 0.0
        mae_sbp = 0.0
        mae_dbp = 0.0
        count = 0
        
        with torch.no_grad():
            for x_sig, x_feat, y_true in val_loader:
                x_sig = x_sig.to(device)
                x_feat = x_feat.to(device)
                y_true = y_true.to(device)
                
                y_pred = model(x_sig, x_feat)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                
                # 计算 MAE (平均绝对误差) - 医生更看重这个指标
                # y_pred[:, 0] 是 SBP, y_pred[:, 1] 是 DBP
                diff = torch.abs(y_pred - y_true)
                mae_sbp += torch.sum(diff[:, 0]).item()
                mae_dbp += torch.sum(diff[:, 1]).item()
                count += y_true.size(0)
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_mae_sbp = mae_sbp / count
        avg_mae_dbp = mae_dbp / count
        
        logging.info(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.2f} | "
            f"Val Loss: {avg_val_loss:.2f} | "
            f"MAE SBP: {avg_mae_sbp:.2f} mmHg | "
            f"MAE DBP: {avg_mae_dbp:.2f} mmHg")
        # === 早停逻辑 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'bp_model_best.pth') # 只保存最好的
            logging.info("Model saved!")
            counter = 0 # 重置计数器
        else:
            counter += 1
            logging.info(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                logging.info("Early stopping triggered!")
                break

    # 保存模型
    torch.save(model.state_dict(), 'bp_model_last.pth')
    logging.info("Training completed. Model saved as 'bp_model_last.pth'")

if __name__ == "__main__":
    # 创建日志目录
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志（同时输出到文件和控制台）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')),
            logging.StreamHandler()
        ]
    )
    main()
