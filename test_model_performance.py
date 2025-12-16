import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from training_on_clean_data import *
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

def load_model_and_predict(model_path, test_data):
    """加载模型并进行预测"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model = ResNetPPG().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # 创建数据集和加载器
    dataset = BPDataset(test_data, is_train=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 存储预测结果和真实值
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x_sig, x_feat, y_true in dataloader:
            x_sig = x_sig.to(device)
            x_feat = x_feat.to(device)
            
            y_pred = model(x_sig, x_feat)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_true.numpy())
    
    # 合并所有批次的结果
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    return predictions, targets

def calculate_metrics(predictions, targets):
    """计算各种评估指标"""
    sbp_pred = predictions[:, 0]
    sbp_true = targets[:, 0]
    dbp_pred = predictions[:, 1]
    dbp_true = targets[:, 1]
    
    metrics = {}
    # 计算绝对误差
    sbp_abs_errors = np.abs(sbp_pred - sbp_true)
    dbp_abs_errors = np.abs(dbp_pred - dbp_true)
    
    # SBP指标=========================================================================================================
    
    metrics['SBP_MAE'] = mean_absolute_error(sbp_true, sbp_pred)
    metrics['SBP_MSE'] = mean_squared_error(sbp_true, sbp_pred)
    metrics['SBP_RMSE'] = np.sqrt(metrics['SBP_MSE'])
    metrics['SBP_R2'] = r2_score(sbp_true, sbp_pred)

    sbp_errors = sbp_pred - sbp_true
    metrics['SBP_ME'] = np.mean(sbp_errors)  # 平均误差（有正负方向）
    metrics['SBP_STD'] = np.std(sbp_errors)   # 标准差
    
    metrics['SBP_Within_5mmHg'] = np.mean(sbp_abs_errors <= 5) * 100
    metrics['SBP_Within_10mmHg'] = np.mean(sbp_abs_errors <= 10) * 100
    metrics['SBP_Within_15mmHg'] = np.mean(sbp_abs_errors <= 15) * 100
    
    # DBP指标=========================================================================================================

    metrics['DBP_MAE'] = mean_absolute_error(dbp_true, dbp_pred)
    metrics['DBP_MSE'] = mean_squared_error(dbp_true, dbp_pred)
    metrics['DBP_RMSE'] = np.sqrt(metrics['DBP_MSE'])
    metrics['DBP_R2'] = r2_score(dbp_true, dbp_pred)
    
    dbp_errors = dbp_pred - dbp_true
    metrics['DBP_ME'] = np.mean(dbp_errors)  # 平均误差（有正负方向）
    metrics['DBP_STD'] = np.std(dbp_errors)   # 标准差
    
    metrics['DBP_Within_5mmHg'] = np.mean(dbp_abs_errors <= 5) * 100
    metrics['DBP_Within_10mmHg'] = np.mean(dbp_abs_errors <= 10) * 100
    metrics['DBP_Within_15mmHg'] = np.mean(dbp_abs_errors <= 15) * 100
    
    # 总体指标=========================================================================================================
    metrics['Overall_MAE'] = (metrics['SBP_MAE'] + metrics['DBP_MAE']) / 2
    metrics['Overall_RMSE'] = (metrics['SBP_RMSE'] + metrics['DBP_RMSE']) / 2
    metrics['Overall_ME'] = (metrics['SBP_ME'] + metrics['DBP_ME']) / 2
    metrics['Overall_STD'] = (metrics['SBP_STD'] + metrics['DBP_STD']) / 2
    
    # 总体误差在5、10、15mmHg以内的比例
    metrics['Overall_Within_5mmHg'] = (metrics['SBP_Within_5mmHg'] + metrics['DBP_Within_5mmHg']) / 2
    metrics['Overall_Within_10mmHg'] = (metrics['SBP_Within_10mmHg'] + metrics['DBP_Within_10mmHg']) / 2
    metrics['Overall_Within_15mmHg'] = (metrics['SBP_Within_15mmHg'] + metrics['DBP_Within_15mmHg']) / 2
    
    return metrics

def plot_scatter_results(predictions, targets):
    """绘制散点图"""
    sbp_pred = predictions[:, 0]
    sbp_true = targets[:, 0]
    dbp_pred = predictions[:, 1]
    dbp_true = targets[:, 1]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SBP散点图
    ax1.scatter(sbp_true, sbp_pred, alpha=0.6, color='blue')
    ax1.plot([sbp_true.min(), sbp_true.max()], [sbp_true.min(), sbp_true.max()], 'r--', lw=2)
    ax1.set_xlabel('True SBP (mmHg)')
    ax1.set_ylabel('Predicted SBP (mmHg)')
    ax1.set_title('SBP Prediction Results Scatter Plot')
    ax1.grid(True, alpha=0.3)
    
    # 添加SBP的MAE和R2信息
    sbp_mae = mean_absolute_error(sbp_true, sbp_pred)
    sbp_r2 = r2_score(sbp_true, sbp_pred)
    ax1.text(0.05, 0.95, f'MAE: {sbp_mae:.2f} mmHg\nR²: {sbp_r2:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # DBP散点图
    ax2.scatter(dbp_true, dbp_pred, alpha=0.6, color='green')
    ax2.plot([dbp_true.min(), dbp_true.max()], [dbp_true.min(), dbp_true.max()], 'r--', lw=2)
    ax2.set_xlabel('True DBP (mmHg)')
    ax2.set_ylabel('Predicted DBP (mmHg)')
    ax2.set_title('DBP Prediction Results Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    # 添加DBP的MAE和R2信息
    dbp_mae = mean_absolute_error(dbp_true, dbp_pred)
    dbp_r2 = r2_score(dbp_true, dbp_pred)
    ax2.text(0.05, 0.95, f'MAE: {dbp_mae:.2f} mmHg\nR²: {dbp_r2:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()


def plot_error_distribution(predictions, targets):
    """绘制误差分布图"""
    sbp_pred = predictions[:, 0]
    sbp_true = targets[:, 0]
    dbp_pred = predictions[:, 1]
    dbp_true = targets[:, 1]
    
    sbp_errors = sbp_pred - sbp_true
    dbp_errors = dbp_pred - dbp_true
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SBP误差分布
    ax1.hist(sbp_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('True SBP - Predicted SBP (mmHg)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('SBP Prediction Error Distribution')
    ax1.grid(True, alpha=0.3)
    
    # DBP误差分布
    ax2.hist(dbp_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('True DBP - Predicted DBP (mmHg)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('DBP Prediction Error Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    

def main():
    """主函数"""
    # 1. 加载测试数据
    print("load test data...")
    all_samples = np.load('clean_ppg_dataset.npy', allow_pickle=True)
    
    # 使用验证集作为测试数据（与训练时相同的划分逻辑）
    subject_ids = list(set([str(sample['subject_id']) for sample in all_samples]))
    subject_ids.sort()
    np.random.seed(42)
    np.random.shuffle(subject_ids)
    
    split_idx = int(len(subject_ids) * 0.8)
    val_ids = subject_ids[:]
    test_data = [s for s in all_samples if str(s['subject_id']) in val_ids]
    
    print(f"number of test samples: {len(test_data)}")
    
    # 2. 加载模型并进行预测
    model_path = "bp_model_best.pth"
    
    predictions, targets = load_model_and_predict(model_path, test_data)
    
    # 3. 计算评估指标
    metrics = calculate_metrics(predictions, targets)
    
    # 4. 打印指标结果
    print("\n=== model performance evaluation results ===")
    print("\n--- SBP metrics ---")
    print(f"SBP_MAE: {metrics['SBP_MAE']:.2f} mmHg")
    print(f"SBP_ME: {metrics['SBP_ME']:.2f} mmHg")
    print(f"SBP_STD: {metrics['SBP_STD']:.2f} mmHg")
    print(f"SBP_RMSE: {metrics['SBP_RMSE']:.2f} mmHg")
    print(f"SBP_R²: {metrics['SBP_R2']:.4f}")
    print(f"SBP ≤5mmHg: {metrics['SBP_Within_5mmHg']:.1f}%")
    print(f"SBP ≤10mmHg: {metrics['SBP_Within_10mmHg']:.1f}%")
    print(f"SBP ≤15mmHg: {metrics['SBP_Within_15mmHg']:.1f}%")
    
    print("\n--- DBP metrics ---")
    print(f"DBP_MAE: {metrics['DBP_MAE']:.2f} mmHg")
    print(f"DBP_ME: {metrics['DBP_ME']:.2f} mmHg")
    print(f"DBP_STD: {metrics['DBP_STD']:.2f} mmHg")
    print(f"DBP_RMSE: {metrics['DBP_RMSE']:.2f} mmHg")
    print(f"DBP_R²: {metrics['DBP_R2']:.4f}")
    print(f"DBP ≤5mmHg: {metrics['DBP_Within_5mmHg']:.1f}%")
    print(f"DBP ≤10mmHg: {metrics['DBP_Within_10mmHg']:.1f}%")
    print(f"DBP ≤15mmHg: {metrics['DBP_Within_15mmHg']:.1f}%")
    
    print("\n--- Overall metrics ---")
    print(f"Overall_MAE: {metrics['Overall_MAE']:.2f} mmHg")
    print(f"Overall_ME: {metrics['Overall_ME']:.2f} mmHg")
    print(f"Overall_STD: {metrics['Overall_STD']:.2f} mmHg")
    print(f"Overall_RMSE: {metrics['Overall_RMSE']:.2f} mmHg")
    print(f"Overall ≤5mmHg: {metrics['Overall_Within_5mmHg']:.1f}%")
    print(f"Overall ≤10mmHg: {metrics['Overall_Within_10mmHg']:.1f}%")
    print(f"Overall ≤15mmHg: {metrics['Overall_Within_15mmHg']:.1f}%")
    # 5. 绘制图表
    plot_scatter_results(predictions, targets)
    plot_error_distribution(predictions, targets)
    plt.show()
    
    print("model performance evaluation completed!")

if __name__ == "__main__":
    main()