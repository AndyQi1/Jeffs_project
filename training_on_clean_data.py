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
from model_config import *
from test_model_performance import *
# ==========================================================================================
def main():
    # 配置日志
    logging.info(f'==========================================================================================')
    logging.info(f'Running training_on_clean_data.py')
    logging.info(f'==========================================================================================')


    # 1. 加载所有数据
    all_samples = np.load('DL_data/clean_ppg_dataset.npy', allow_pickle=True)

    # 2. 提取所有 subject_id 以便进行划分
    # 注意：你的 subject_id 可能是字符串或数字，这里确保统一
    subject_ids = list(set([str(sample['subject_id']) for sample in all_samples]))
    subject_ids.sort()

    # 3. 按人头划分 (80% 训练, 20% 验证)
    np.random.seed(42)
    np.random.shuffle(subject_ids)

    split_idx = int(len(subject_ids) * 0.8)
    train_ids = subject_ids[:split_idx]
    val_ids = subject_ids[split_idx:]

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
    train_dataset = BPDataset(train_data, is_train=True)
    val_dataset = BPDataset(val_data, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 模型、损失、优化器
    model = ResNetPPG().to(device)
    criterion = nn.MSELoss() # 回归任务常用均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
    )

    # --- 开始训练 ---
    epochs = 50
    best_val_loss = float('inf')
    patience = 10  # 容忍多少轮不下降
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
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_para/bp_model_best.pth') # 只保存最好的
            logging.info("Best Model saved!")
            counter = 0 # 重置计数器
        else:
            counter += 1
            logging.info(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                logging.info("Early stopping triggered!")
                break

    # 保存模型
    torch.save(model.state_dict(), 'model_para/bp_model_last.pth')
    logging.info("Training completed. Last Model saved as 'bp_model_last.pth'")

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
    test_model()
