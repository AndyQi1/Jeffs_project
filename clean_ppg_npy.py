# 基于raw_ppg_dataset.npy文件，提取clean ppg分段数据
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


# 读取npy文件
dataset = np.load('raw_ppg_dataset.npy', allow_pickle=True)

# 查看数据结构
print(f"数据类型: {type(dataset)}")
print(f"数据形状: {dataset.shape}")
print(f"样本数量: {len(dataset)}")

segmented_ppg = []

for sample in dataset:
    ppg_distal = sample['ppg_distal']
    ppg_proximal = sample['ppg_proximal']
    seconds = np.arange(0, len(ppg_distal)/500, 1/500)
    fs = 500  # 计算采样率
    window_size = fs*15  # 15秒窗口
    overlap_size = 5*fs  # 5秒重叠
    step_size = window_size - overlap_size

    # signal processing ==========================================================================================

    # remove linear trend - detrend
    ppg_distal_detrended = remove_linear_trend(ppg_distal)
    ppg_proximal_detrended = remove_linear_trend(ppg_proximal)

    # bandpass filter - remove noise
    ppg_distal_filtered = butter_bandpass_filter(ppg_distal_detrended, 0.4, 8, fs)
    ppg_proximal_filtered = butter_bandpass_filter(ppg_proximal_detrended, 0.4, 8, fs)

    # Hampel filter - remove outliers
    ppg_distal_filtered, _ = hampel_filter(ppg_distal_filtered)
    ppg_proximal_filtered, _ = hampel_filter(ppg_proximal_filtered)

    # Z-score normalization
    ppg_distal_norm = (ppg_distal_filtered - np.mean(ppg_distal_filtered)) / np.std(ppg_distal_filtered)
    ppg_proximal_norm = (ppg_proximal_filtered - np.mean(ppg_proximal_filtered)) / np.std(ppg_proximal_filtered)

    # sliding window ============================================================================================
    segmented_data_list = []
    for start in range(0, len(ppg_distal_norm) - window_size + 1, step_size):
        end = start + window_size
        ppg_distal_window = ppg_distal_norm[start:end]
        ppg_proximal_window = ppg_proximal_norm[start:end]

        segmented_data = sample.copy()
        segmented_data['ppg_distal'] = ppg_distal_window
        segmented_data['ppg_proximal'] = ppg_proximal_window

        segmented_data_list.append(segmented_data)
    print(f"样本 {sample['subject_id']}的活动类型{sample['activity']}分割完成，共 {len(segmented_data_list)} 个窗口")

    segmented_ppg.extend(segmented_data_list)

np.save('clean_ppg_dataset.npy', np.array(segmented_ppg))
print(f"分段完成，共 {len(segmented_ppg)} 个窗口")

    

