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
dataset = np.load('DL_data/raw_ppg_dataset.npy', allow_pickle=True)

# 查看数据结构
print(f"数据类型: {type(dataset)}")
print(f"数据形状: {dataset.shape}")
print(f"样本数量: {len(dataset)}")

segmented_ppg = []

for sample in dataset:
    ppg_distal = sample['ppg_distal']
    ppg_proximal = sample['ppg_proximal']

    fs = 500  # 计算采样率
    
    segment_length = 10  # 10秒窗口

    # signal processing ==========================================================================================
    PTT_list, HR_list, PWV_list, SVRI_list, IPA_list, skew_list, \
        similarity_list, perfusion_index_list, ppg_distal_waveforms, ppg_proximal_waveforms = extract_ppg_features(
        ppg_distal, ppg_proximal, segment_length, fs=fs)

    # sliding window ============================================================================================
    segmented_data_list = []

    segmented_data = sample.copy()
    segmented_data['ppg_distal_waveforms'] = ppg_distal_waveforms
    segmented_data['ppg_proximal_waveforms'] = ppg_proximal_waveforms
    segmented_data['PTT_list'] = PTT_list
    segmented_data['HR_list'] = HR_list
    segmented_data['PWV_list'] = PWV_list
    segmented_data['SVRI_list'] = SVRI_list
    segmented_data['IPA_list'] = IPA_list
    segmented_data['skew_list'] = skew_list
    segmented_data['similarity_list'] = similarity_list
    segmented_data['perfusion_index_list'] = perfusion_index_list

    segmented_data_list.append(segmented_data)
    print(f"样本 {sample['subject_id']}的活动类型{sample['activity']}分割完成，共 {len(segmented_data_list)} 个窗口")

    segmented_ppg.extend(segmented_data_list)

np.save('DL_data/clean_ppg_dataset.npy', np.array(segmented_ppg))
print(f"分段完成，共 {len(segmented_ppg)} 个窗口")

    

