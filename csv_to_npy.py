# 从CSV文件中读取数据存储为npy文件
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import os

plt.rcParams['font.family'] = 'Times New Roman'


# 设置目录路径
# 使用 r'' (raw string) 来避免 Windows 反斜杠 \ 被转义的问题
folder_path = r'PTT_data\physionet.org\pulse-transit-time-ppg\csv'

# 获取该目录下所有 csv 文件的路径
file_pattern = os.path.join(folder_path, '*.csv')
csv_files = glob.glob(file_pattern)

print(f"在目录中找到了 {len(csv_files)} 个 CSV 文件。")

## ==========================================================================================================
# 提取文件中的性别、身高、体重、年龄、心率
subject_info_file = 'subjects_info.csv'
file_path = os.path.join(folder_path, subject_info_file)
df = pd.read_csv(file_path)
gender = df[df['activity'] == 'walk']['gender'].reset_index(drop=True)
height = df[df['activity'] == 'walk']['height'].reset_index(drop=True)
weight = df[df['activity'] == 'walk']['weight'].reset_index(drop=True)
age = df[df['activity'] == 'walk']['age'].reset_index(drop=True)
walk_HR = 1/2 * (df[df['activity'] == 'walk']['hr_1_start'] + df[df['activity'] == 'walk']['hr_1_end']).reset_index(drop=True)
walk_SBP = 1/2 * (df[df['activity'] == 'walk']['bp_sys_start'] + df[df['activity'] == 'walk']['bp_sys_end']).reset_index(drop=True)
walk_DBP = 1/2 * (df[df['activity'] == 'walk']['bp_dia_start'] + df[df['activity'] == 'walk']['bp_dia_end']).reset_index(drop=True)

run_HR = 1/2 * (df[df['activity'] == 'run']['hr_1_start'] + df[df['activity'] == 'run']['hr_1_end']).reset_index(drop=True)
run_SBP = 1/2 * (df[df['activity'] == 'run']['bp_sys_start'] + df[df['activity'] == 'run']['bp_sys_end']).reset_index(drop=True)
run_DBP = 1/2 * (df[df['activity'] == 'run']['bp_dia_start'] + df[df['activity'] == 'run']['bp_dia_end']).reset_index(drop=True)

sit_HR = 1/2 * (df[df['activity'] == 'sit']['hr_1_start'] + df[df['activity'] == 'sit']['hr_1_end']).reset_index(drop=True)
sit_SBP = 1/2 * (df[df['activity'] == 'sit']['bp_sys_start'] + df[df['activity'] == 'sit']['bp_sys_end']).reset_index(drop=True)
sit_DBP = 1/2 * (df[df['activity'] == 'sit']['bp_dia_start'] + df[df['activity'] == 'sit']['bp_dia_end']).reset_index(drop=True)

print(f'subject信息提取完成。')

## ==========================================================================================================
# 提取文件中的PPG数据
dataset = []
activity_map = {'walk': 1, 'run': 2, 'sit': 3}
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    if 'subjects_info' in file_name:
        continue   
    subject_id, activity = file_name.split('_')[:2]
    subject_id = subject_id.replace('s', '')
    activity = activity.replace('.csv', '')
    print(f'subject_id: {subject_id}, activity: {activity}')
    df = pd.read_csv(file_path)
    ppg_distal = df['pleth_3']
    ppg_proximal = df['pleth_6']
    if activity == 'walk':
        HR = walk_HR[int(subject_id)-1]
        SBP = walk_SBP[int(subject_id)-1]
        DBP = walk_DBP[int(subject_id)-1]
    elif activity == 'run':
        HR = run_HR[int(subject_id)-1]
        SBP = run_SBP[int(subject_id)-1]
        DBP = run_DBP[int(subject_id)-1]
    elif activity == 'sit':
        HR = sit_HR[int(subject_id)-1]
        SBP = sit_SBP[int(subject_id)-1]
        DBP = sit_DBP[int(subject_id)-1]
    else:
        raise ValueError(f"Unknown activity: {activity}")
    data_dict = {
        'subject_id': subject_id,           # 输入：subject_id
        'ppg_distal': ppg_distal,           # 输入：远端ppg信号
        'ppg_proximal': ppg_proximal,       # 输入：近端ppg信号
        'HR': HR,                           # 输入4: 心率
        'SBP': SBP,                           # 输入5: 收缩压
        'DBP': DBP,                           # 输入6: 舒张压
        'gender': gender[int(subject_id)-1],       # 输入：性别
        'activity': activity_map[activity],     # 输入: 活动类型
        'height': height[int(subject_id)-1],       # 标签: 身高
        'weight': weight[int(subject_id)-1],       # 标签: 体重
        'age': age[int(subject_id)-1],       # 标签: 年龄
    }
    dataset.append(data_dict)


np.save('DL_data/raw_ppg_dataset.npy', dataset)

print(f"成功打包 {len(dataset)} 个样本。")


