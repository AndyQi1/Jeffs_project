import pandas as pd
import glob
import os
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# 设置目录路径
# 使用 r'' (raw string) 来避免 Windows 反斜杠 \ 被转义的问题
folder_path = r'PTT_data\physionet.org\pulse-transit-time-ppg\csv'

# 获取该目录下所有 csv 文件的路径
file_pattern = os.path.join(folder_path, '*.csv')
csv_files = glob.glob(file_pattern)

print(f"在目录中找到了 {len(csv_files)} 个 CSV 文件。")

## ==========================================================================================================
# 分析文件中的性别、身高、体重、年龄、心率
subject_info_file = 'subjects_info.csv'

for file_path in csv_files:
    if subject_info_file in file_path:
        df = pd.read_csv(file_path)
        gender = df[df['activity'] == 'walk']['gender']
        height = df[df['activity'] == 'walk']['height']
        weight = df[df['activity'] == 'walk']['weight']
        age = df[df['activity'] == 'walk']['age']
        walk_HR = 1/2 * (df[df['activity'] == 'walk']['hr_1_start'] + df[df['activity'] == 'walk']['hr_1_end'])
        walk_SBP = 1/2 * (df[df['activity'] == 'walk']['bp_sys_start'] + df[df['activity'] == 'walk']['bp_sys_end'])
        walk_DBP = 1/2 * (df[df['activity'] == 'walk']['bp_dia_start'] + df[df['activity'] == 'walk']['bp_dia_end'])

        run_HR = 1/2 * (df[df['activity'] == 'run']['hr_1_start'] + df[df['activity'] == 'run']['hr_1_end'])
        run_SBP = 1/2 * (df[df['activity'] == 'run']['bp_sys_start'] + df[df['activity'] == 'run']['bp_sys_end'])
        run_DBP = 1/2 * (df[df['activity'] == 'run']['bp_dia_start'] + df[df['activity'] == 'run']['bp_dia_end'])

        sit_HR = 1/2 * (df[df['activity'] == 'sit']['hr_1_start'] + df[df['activity'] == 'sit']['hr_1_end'])
        sit_SBP = 1/2 * (df[df['activity'] == 'sit']['bp_sys_start'] + df[df['activity'] == 'sit']['bp_sys_end'])
        sit_DBP = 1/2 * (df[df['activity'] == 'sit']['bp_dia_start'] + df[df['activity'] == 'sit']['bp_dia_end'])
        break

## ===========================================================================================================
# 图形化展示dataset数据构成
male_num = sum(gender == 'male')
female_num = sum(gender == 'female')
# 绘制性别比例柱状图 （gender）
plt.figure(figsize=(8, 6))
plt.pie([male_num, female_num], labels = ['Male', 'Female'], autopct='%1.1f%%')
plt.title('Gender Distribution in the Dataset')

# 绘制线图带误差棒（HR）
plt.figure(figsize=(8, 6))
plt.grid(True, alpha=0.3)
activities = ['walk', 'run', 'sit']
HR_means = [walk_HR.mean(), run_HR.mean(), sit_HR.mean()]
HR_stds = [walk_HR.std(), run_HR.std(), sit_HR.std()]


plt.errorbar(activities, HR_means, yerr=HR_stds, fmt='-o', capsize=5, markersize=8)
plt.ylabel('Heart Rate (bpm)')
plt.title('Heart Rate by Activity')
plt.ylim(40, 100)

SBP_means = [walk_SBP.mean(), run_SBP.mean(), sit_SBP.mean()]
SBP_stds = [walk_SBP.std(), run_SBP.std(), sit_SBP.std()]

DBP_means = [walk_DBP.mean(), run_DBP.mean(), sit_DBP.mean()]
DBP_stds = [walk_DBP.std(), run_DBP.std(), sit_DBP.std()]

# 绘制线图带误差棒（SBP、DBP）
fig, ax1 = plt.subplots()
plt.grid(True, alpha=0.3)
# 左纵轴 - SBP
ax1.errorbar(activities, SBP_means, yerr=SBP_stds, fmt='o-', color='red', 
             label='SBP', capsize=5)
ax1.set_ylabel('SBP (mmHg)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# 右纵轴 - DBP
ax2 = ax1.twinx()
ax2.errorbar(activities, DBP_means, yerr=DBP_stds, fmt='s-', color='blue', 
             label='DBP', capsize=5)
ax2.set_ylabel('DBP (mmHg)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Blood Pressure by Activity')
# 左纵轴：固定为80-160
ax1.set_ylim(80, 160)

# 右纵轴：固定为40-100  
ax2.set_ylim(40, 100)


# 绘制柱状分布图（height、weight、age、SBP、DBP、HR）
plt.figure(figsize=(12, 8))

# 身高
plt.subplot(2, 3, 1)
plt.hist(height, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Height (m)')
plt.ylabel('Frequency')
plt.title('Height Distribution')
plt.grid(True, alpha=0.3)

# 体重
plt.subplot(2, 3, 2)
plt.hist(weight, bins=10, color='salmon', edgecolor='black', alpha=0.7)
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.title('Weight Distribution')
plt.grid(True, alpha=0.3)

# 年龄
plt.subplot(2, 3, 3)
plt.hist(age, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.grid(True, alpha=0.3)

# SBP
plt.subplot(2, 3, 4)
all_SBP = pd.concat([walk_SBP, run_SBP, sit_SBP], ignore_index=True)
plt.hist(all_SBP, bins=10, color='red', edgecolor='black', alpha=0.7)
plt.xlabel('SBP (mmHg)')
plt.ylabel('Frequency')
plt.title('SBP Distribution')
plt.grid(True, alpha=0.3)

# DBP
plt.subplot(2, 3, 5)
all_DBP = pd.concat([walk_DBP, run_DBP, sit_DBP], ignore_index=True)
plt.hist(all_DBP, bins=10, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('DBP (mmHg)')
plt.ylabel('Frequency')
plt.title('DBP Distribution')
plt.grid(True, alpha=0.3)

# HR
plt.subplot(2, 3, 6)
all_HR = pd.concat([walk_HR, run_HR, sit_HR], ignore_index=True)
plt.hist(all_HR, bins=10, color='orange', edgecolor='black', alpha=0.7)
plt.xlabel('HR (bpm)')
plt.ylabel('Frequency')
plt.title('HR Distribution')
plt.grid(True, alpha=0.3)



# 在plt.show()之前添加
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)  # 调整垂直间距
plt.show()

