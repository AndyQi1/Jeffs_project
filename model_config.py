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
    def __init__(self, nested_data_list, is_train=True):
        """
        nested_data_list: 你的新数据结构，包含 segmented_data_list
        """
        self.is_train = is_train
        self.samples = [] # 用来存打平后的数据

        dropped_count = 0 # 计数器：记录丢弃了多少条数据
        
        # --- 1. 在初始化时将嵌套列表“打平” ---
        for subject_data in nested_data_list:
            # 获取波形列表的长度 (假设所有list长度一致)
            num_windows = len(subject_data['ppg_distal_waveforms'])
            
            # 提取公共静态信息 (整个样本通用的信息)
            static_info = {
                'subject_id': subject_data.get('subject_id'),
                'gender': subject_data['gender'],
                'activity': subject_data['activity'],
                'height': subject_data['height'],
                'weight': subject_data['weight'],
                'age': subject_data['age'],
                'HR': subject_data['HR'],
                'SBP': subject_data['SBP'],
                'DBP': subject_data['DBP']
            }
            
            # 遍历每一个 10s 窗口
            for i in range(num_windows):
                # 创建一个独立的训练样本
                flat_item = static_info.copy()
                
                # 提取动态信息 (列表中的第 i 个元素)
                # 顺便解决 np.float64 问题，用 .item() 转为 python float
                flat_item['ppg_distal'] = subject_data['ppg_distal_waveforms'][i]["ppg_signal"]
                flat_item['ppg_proximal'] = subject_data['ppg_proximal_waveforms'][i]["ppg_signal"]
                
                # 提取特征列表中的对应值
                # flat_item['HR'] = subject_data['HR_list'][i]
                flat_item["PTT"] = subject_data['PTT_list'][i]
                flat_item["PWV"] = subject_data['PWV_list'][i]
                flat_item["skew"] = subject_data['skew_list'][i]
                flat_item["SVRI"] = subject_data['SVRI_list'][i]
                flat_item["IPA"] = subject_data['IPA_list'][i]
                flat_item["similarity"] = subject_data['similarity_list'][i]                
                flat_item["perfusion_index"] = subject_data['perfusion_index_list'][i]
                if (np.isnan(flat_item["HR"]) or np.isnan(flat_item["PTT"]) or 
                    np.isnan(flat_item["PWV"]) or np.isnan(flat_item["skew"]) or 
                    np.isnan(flat_item["SVRI"]) or np.isnan(flat_item["IPA"]) or 
                    np.isnan(flat_item["similarity"]) or np.isnan(flat_item["perfusion_index"])):
                    dropped_count += 1
                    continue
                
                self.samples.append(flat_item)
        print(f"数据集初始化完成：保留样本 {len(self.samples)} 个，丢弃样本 {dropped_count} 个。")
                
    def __len__(self):
        # 现在返回的是打平后的总窗口数
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 直接拿打平后的数据
        item = self.samples[idx]
        
        # --- 1. 处理时序信号 (双通道) ---
        p_dist = item['ppg_distal']
        p_prox = item['ppg_proximal']
        
        # 确保是 float32 类型的 numpy 数组
        if not isinstance(p_dist, np.ndarray):
            p_dist = np.array(p_dist)
        if not isinstance(p_prox, np.ndarray):
            p_prox = np.array(p_prox)
            
        p_dist = p_dist.astype(np.float32)
        p_prox = p_prox.astype(np.float32)
        
        # 归一化 (Min-Max)
        p_dist = (p_dist - p_dist.min()) / (p_dist.max() - p_dist.min() + 1e-6)
        p_prox = (p_prox - p_prox.min()) / (p_prox.max() - p_prox.min() + 1e-6)
        
        # 堆叠 (2, Length)
        signals = np.stack([p_dist, p_prox], axis=0) 
        
        if self.is_train:
            # 数据增强逻辑保持不变
            scale = np.random.uniform(0.9, 1.1)
            signals = signals * scale
            noise = np.random.normal(0, 0.01, signals.shape)
            signals = signals + noise
            shift = np.random.randint(-10, 10)
            signals = np.roll(signals, shift, axis=1)
            
        x_signal = torch.tensor(signals, dtype=torch.float32)
        
        # --- 2. 处理标量特征 ---
        gender_mapping = {'male': 1, 'female': 0, 'Male': 1, 'Female': 0}

        # 处理 np.float64 或 list 元素
        def to_float(val):
            if hasattr(val, 'item'): return val.item() # 处理 numpy scalar
            return float(val)

        hr = to_float(item['HR']) / 200.0
        gen = float(gender_mapping.get(item['gender'], 0)) # 加个 .get 防报错
        act = to_float(item['activity']) / 3.0    
        h = to_float(item['height']) / 200.0      
        w = to_float(item['weight']) / 150.0      
        age = to_float(item['age']) / 100.0
        ptt = to_float(item['PTT'])
        pwv = to_float(item['PWV'])
        skew = to_float(item['skew'])
        svri = to_float(item['SVRI'])
        ipa = to_float(item['IPA'])
        similarity = to_float(item['similarity'])
        perfusion_index = to_float(item['perfusion_index'])
        
        features = np.array([hr, gen, act, h, w, age, ptt, pwv, skew, svri, ipa, similarity, perfusion_index], dtype=np.float32)
        x_feat = torch.tensor(features, dtype=torch.float32)
        
        # --- 3. 处理标签 ---
        sbp = to_float(item['SBP'])
        dbp = to_float(item['DBP'])
        
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
            nn.Linear(6 + 7, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU()
        )
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
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
        combined = aux + x
        
        return self.regressor(combined)