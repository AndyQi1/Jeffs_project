import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
import os
print("当前工作目录:", os.getcwd())

# 假设你的 BPDataset 类定义在上面，或者从你的文件中 import 进来
from training_on_clean_data import BPDataset 

def analyze_feature_distribution():
    # 1. 加载数据
    data_path = 'DL_data/clean_ppg_dataset.npy' # 请修改为你的实际路径
    print(f"正在加载数据: {data_path} ...")
    all_samples = np.load(data_path, allow_pickle=True)
    
    # 2. 实例化 Dataset
    # 注意：is_train=False 即可，因为我们要看的是原始数据，不需要由 __getitem__ 做增强
    print("正在初始化 Dataset 进行数据清洗...")
    dataset = BPDataset(all_samples, is_train=False)
    
    # 3. 核心步骤：直接获取清洗后的样本列表
    # dataset.samples 是你在 __init__ 里 append 出来的列表，包含原始数值的字典
    raw_data_list = dataset.samples
    
    print(f"提取完成，共有 {len(raw_data_list)} 个有效样本。")
    
    # 4. 转换为 Pandas DataFrame (神器，方便分析)
    # 这里的 keys 对应你在 __init__ 里 flat_item 的 keys
    df = pd.DataFrame(raw_data_list)
    
    # 这里的列名应该是: ['subject_id', 'gender', ..., 'HR', 'PTT', 'PWV', 'skew', ...]
    # 我们只关心你想看的特征列
    feature_cols = ['PTT', 'PWV', 'skew', 'SVRI', 'IPA', 'similarity', 'perfusion_index', 'HR', 'SBP', 'DBP']
    
    # 确保数据类型是 float (处理可能存在的 np.float64 对象)
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # 5. 可视化分析
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    
    # --- A. 打印统计摘要 (Mean, Std, Min, Max) ---
    print("\n=== 数据统计摘要 ===")
    print(df[feature_cols].describe())
    
    # --- B. 绘制直方图 (Histograms) ---
    # 看看特征是否符合正态分布，有没有离群值
    num_features = len(feature_cols)
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    for i, col in enumerate(feature_cols):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col], kde=True, bins=50) # kde=True 显示密度曲线
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # --- C. 绘制箱线图 (Boxplots) ---
    # 专门用来抓离群值 (Outliers)
    plt.figure(figsize=(15, 6))
    # 因为不同特征量纲差异大（SVRI是几千，PTT是几百，IPA是几），建议标准化后再画，或者分开画
    # 这里演示单独画几个关键的
    subset_cols = ['PTT', 'HR', 'PWV'] 
    sns.boxplot(data=df[subset_cols])
    plt.title('Boxplot of PTT, HR, PWV')
    plt.show()

    # --- D. 相关性热力图 (Correlation Heatmap) ---
    # 看看哪些特征和 SBP/DBP 高度相关
    plt.figure(figsize=(12, 10))
    corr_matrix = df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    analyze_feature_distribution()