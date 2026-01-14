from function_lib import *
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 引入模型配置
from model_config import * 

# ==========================================================================================
def train_one_fold(train_data, val_data, fold_idx, device, log_dir):
    """训练单个 Fold 的模型"""
    
    # 数据加载器
    train_dataset = BPDataset(train_data, is_train=True)
    val_dataset = BPDataset(val_data, is_train=False)
    
    # 增大 BatchSize 利用 3060 显卡
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型 (每个 Fold 都要重新初始化，不能继承上一次的权重！)
    model = ResNetPPG().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    epochs = 50 # 可以根据需要调整
    patience = 10
    best_val_loss = float('inf')
    counter = 0
    best_model_path = os.path.join(log_dir, f'fold_{fold_idx}_best.pth')
    
    logging.info(f"--- Starting Fold {fold_idx} Training ---")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for x_sig, x_feat, y_true in train_loader:
            x_sig, x_feat, y_true = x_sig.to(device), x_feat.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x_sig, x_feat)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证 (用于早停)
        model.eval()
        val_loss = 0.0
        val_mae_sbp = 0.0
        val_mae_dbp = 0.0
        val_count = 0
        
        with torch.no_grad():
            for x_sig, x_feat, y_true in val_loader:
                x_sig, x_feat, y_true = x_sig.to(device), x_feat.to(device), y_true.to(device)
                y_pred = model(x_sig, x_feat)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()
                
                # 计算 MAE 用于显示
                diff = torch.abs(y_pred - y_true)
                val_mae_sbp += torch.sum(diff[:, 0]).item()
                val_mae_dbp += torch.sum(diff[:, 1]).item()
                val_count += y_true.size(0)
        
        # 防止除以零
        if len(train_loader) > 0:
            avg_train_loss = train_loss / len(train_loader)
        else:
            avg_train_loss = 0
            
        if len(val_loader) > 0 and val_count > 0:
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae_sbp = val_mae_sbp / val_count
            avg_val_mae_dbp = val_mae_dbp / val_count
        else:
            avg_val_loss = float('inf')
            avg_val_mae_sbp = 0
            avg_val_mae_dbp = 0
        
        # === 【这里就是你要的打印】 ===
        # 打印当前 Epoch 的情况
        logging.info(f"Fold {fold_idx} | Ep {epoch+1}/{epochs} | "
                     f"TrainLoss: {avg_train_loss:.1f} | "
                     f"ValLoss: {avg_val_loss:.1f} | "
                     f"Val MAE: SBP={avg_val_mae_sbp:.2f} DBP={avg_val_mae_dbp:.2f}")
        
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Fold {fold_idx}: Early stopping at epoch {epoch+1}")
                break
                
    # 加载本 Fold 最好的模型返回
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model

# ==========================================================================================
# 辅助函数：预测单折 (Predict One Fold)
# ==========================================================================================
def predict_one_fold(model, test_data, device):
    """在测试集(被留下的那个人)上进行预测"""
    dataset = BPDataset(test_data, is_train=False)
    # Batch size 无所谓，只是为了推理
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model.eval()
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for x_sig, x_feat, y_true in dataloader:
            x_sig = x_sig.to(device)
            x_feat = x_feat.to(device)
            
            y_pred = model(x_sig, x_feat)
            
            preds_list.append(y_pred.cpu().numpy())
            targets_list.append(y_true.numpy())
            
    if len(preds_list) == 0:
        return np.array([]), np.array([])
        
    return np.vstack(preds_list), np.vstack(targets_list)

# ==========================================================================================
# 主流程：留一法交叉验证 (LOSO Main Loop)
# ==========================================================================================
def run_loso_cv():
    # 1. 配置日志
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'loso_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting Leave-One-Subject-Out (LOSO) Cross-Validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. 加载全量数据
    all_samples = np.load('DL_data/clean_ppg_dataset.npy', allow_pickle=True)
    
    # 提取所有受试者 ID
    subject_ids = sorted(list(set([str(sample['subject_id']) for sample in all_samples])))
    n_subjects = len(subject_ids)
    logging.info(f"Total subjects: {n_subjects}")
    
    # 用于存储所有 Folds 的汇总结果
    global_predictions = []
    global_targets = []
    
    # === LOSO 循环开始 ===
    for idx, test_sub_id in enumerate(subject_ids):
        logging.info(f"===========================================================")
        logging.info(f"Processing Fold {idx+1}/{n_subjects} | Test Subject: {test_sub_id}")
        logging.info(f"===========================================================")
        
        # A. 数据划分
        # 测试集：当前这 1 个人
        test_data = [s for s in all_samples if str(s['subject_id']) == test_sub_id]
        
        # 剩余池：剩下的 N-1 个人
        remaining_data = [s for s in all_samples if str(s['subject_id']) != test_sub_id]
        
        # 从剩余池中，再划分出训练集和内部验证集 (用于 Early Stopping)
        # 比如：剩下的 N-1 个人里，80% 训练，20% 做内部验证
        remaining_ids = sorted(list(set([str(s['subject_id']) for s in remaining_data])))
        np.random.seed(42 + idx) # 每一折变个种子，增加随机性
        np.random.shuffle(remaining_ids)
        
        split_point = int(len(remaining_ids) * 0.8)
        train_sub_ids = remaining_ids[:split_point]
        val_sub_ids = remaining_ids[split_point:] # 内部验证集
        
        train_data = [s for s in remaining_data if str(s['subject_id']) in train_sub_ids]
        val_data = [s for s in remaining_data if str(s['subject_id']) in val_sub_ids]
        
        logging.info(f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Test samples: {len(test_data)}")
        
        # B. 训练这一折的模型
        # 传入 log_dir 用于保存临时的 best model
        model = train_one_fold(train_data, val_data, idx, device, log_dir)
        
        # C. 在测试集(被留下的那个人)上预测
        fold_preds, fold_targets = predict_one_fold(model, test_data, device)
        
        # D. 收集结果
        if len(fold_preds) > 0:
            global_predictions.append(fold_preds)
            global_targets.append(fold_targets)
            
            # 计算单折的简单指标打印看看
            mae_sbp = mean_absolute_error(fold_targets[:, 0], fold_preds[:, 0])
            mae_dbp = mean_absolute_error(fold_targets[:, 1], fold_preds[:, 1])
            logging.info(f"Fold {idx+1} Result: MAE SBP={mae_sbp:.2f}, MAE DBP={mae_dbp:.2f}")
        else:
            logging.warning(f"Fold {idx+1} (Subject {test_sub_id}) produced no predictions (maybe NaN drop).")

    # === 循环结束，计算总指标 ===
    logging.info("===========================================================")
    logging.info("LOSO CV Completed. Calculating Global Metrics...")
    
    # 拼接所有结果
    final_preds = np.vstack(global_predictions)
    final_targets = np.vstack(global_targets)
    
    # 调用之前的评估函数
    metrics = calculate_metrics(final_preds, final_targets)
    
    # 打印最终结果
    log_final_metrics(metrics)
    
    # 绘图
    plot_scatter_results(final_preds, final_targets)
    plot_error_distribution(final_preds, final_targets)
    plt.show()

# ==========================================================================================
# 指标计算与绘图函数 (保持原样，稍微封装一下)
# ==========================================================================================
def calculate_metrics(predictions, targets):
    sbp_pred, sbp_true = predictions[:, 0], targets[:, 0]
    dbp_pred, dbp_true = predictions[:, 1], targets[:, 1]
    
    metrics = {}
    # SBP
    metrics['SBP_MAE'] = mean_absolute_error(sbp_true, sbp_pred)
    metrics['SBP_RMSE'] = np.sqrt(mean_squared_error(sbp_true, sbp_pred))
    metrics['SBP_R2'] = r2_score(sbp_true, sbp_pred)
    sbp_err = np.abs(sbp_pred - sbp_true)
    metrics['SBP_Within_5'] = np.mean(sbp_err <= 5) * 100
    metrics['SBP_Within_10'] = np.mean(sbp_err <= 10) * 100
    metrics['SBP_Within_15'] = np.mean(sbp_err <= 15) * 100
    
    # DBP
    metrics['DBP_MAE'] = mean_absolute_error(dbp_true, dbp_pred)
    metrics['DBP_RMSE'] = np.sqrt(mean_squared_error(dbp_true, dbp_pred))
    metrics['DBP_R2'] = r2_score(dbp_true, dbp_pred)
    dbp_err = np.abs(dbp_pred - dbp_true)
    metrics['DBP_Within_5'] = np.mean(dbp_err <= 5) * 100
    metrics['DBP_Within_10'] = np.mean(dbp_err <= 10) * 100
    metrics['DBP_Within_15'] = np.mean(dbp_err <= 15) * 100
    
    return metrics

def log_final_metrics(metrics):
    logging.info("\n=== Final LOSO Performance Evaluation ===")
    logging.info(f"SBP MAE: {metrics['SBP_MAE']:.2f} | RMSE: {metrics['SBP_RMSE']:.2f} | R2: {metrics['SBP_R2']:.3f}")
    logging.info(f"SBP Accuracy: <5mmHg: {metrics['SBP_Within_5']:.1f}% | <10mmHg: {metrics['SBP_Within_10']:.1f}%")
    
    logging.info(f"DBP MAE: {metrics['DBP_MAE']:.2f} | RMSE: {metrics['DBP_RMSE']:.2f} | R2: {metrics['DBP_R2']:.3f}")
    logging.info(f"DBP Accuracy: <5mmHg: {metrics['DBP_Within_5']:.1f}% | <10mmHg: {metrics['DBP_Within_10']:.1f}%")

def plot_scatter_results(predictions, targets):
    # (此处直接复制你之前的绘图代码即可)
    sbp_pred = predictions[:, 0]
    sbp_true = targets[:, 0]
    dbp_pred = predictions[:, 1]
    dbp_true = targets[:, 1]
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SBP
    ax1.scatter(sbp_true, sbp_pred, alpha=0.5, color='blue', s=10)
    ax1.plot([sbp_true.min(), sbp_true.max()], [sbp_true.min(), sbp_true.max()], 'r--', lw=2)
    ax1.set_title(f"All Subjects SBP (LOSO)")
    ax1.set_xlabel('Reference SBP (mmHg)')
    ax1.set_ylabel('Predicted SBP (mmHg)')
    
    # DBP
    ax2.scatter(dbp_true, dbp_pred, alpha=0.5, color='green', s=10)
    ax2.plot([dbp_true.min(), dbp_true.max()], [dbp_true.min(), dbp_true.max()], 'r--', lw=2)
    ax2.set_title(f"All Subjects DBP (LOSO)")
    ax2.set_xlabel('Reference DBP (mmHg)')
    
    plt.tight_layout()

def plot_error_distribution(predictions, targets):
    # (此处直接复制你之前的绘图代码即可)
    sbp_err = predictions[:, 0] - targets[:, 0]
    dbp_err = predictions[:, 1] - targets[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.hist(sbp_err, bins=50, color='blue', alpha=0.7, density=True)
    ax1.set_title("SBP Error Distribution")
    ax2.hist(dbp_err, bins=50, color='green', alpha=0.7, density=True)
    ax2.set_title("DBP Error Distribution")
    plt.tight_layout()

if __name__ == "__main__":
    run_loso_cv()