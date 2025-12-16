import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from sklearn.linear_model import LinearRegression
import glob

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def hampel_filter(data, window_size=5, n_sigma=3):
    """
    Hampel滤波器实现：检测并修正时序数据中的异常值
    参数：
        data: 输入时序数据（1D numpy数组或列表）
        window_size: 滑动窗口大小（必须为奇数，推荐5-11）
        n_sigma: 异常值判断的倍数（推荐3，即3倍MAD范围）
    返回：
        filtered_data: 滤波后的数据
        outlier_indices: 异常值的索引列表
    """
    # 输入数据转为numpy数组
    data = np.array(data, dtype=float)
    filtered_data = data.copy()
    outlier_indices = []
    n = len(data)
    
    # 窗口大小必须为奇数，否则调整
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2  # 窗口半宽（如窗口5，半宽2）

    # 遍历每个数据点（跳过窗口无法覆盖的边缘点）
    for i in range(half_window, n - half_window):
        # 1. 取当前点的滑动窗口
        window = data[i - half_window : i + half_window + 1]
        
        # 2. 计算窗口内的中位数和MAD
        median = np.median(window)
        mad = np.median(np.abs(window - median))  # 中位数绝对偏差
        
        # 3. 计算异常值判断阈值（1.4826是正态分布下MAD→标准差的系数）
        threshold = n_sigma * 1.4826 * mad
        
        # 4. 判断当前点是否为异常值
        if np.abs(data[i] - median) > threshold:
            # 异常值：用中位数替换
            filtered_data[i] = median
            outlier_indices.append(i)

    return filtered_data, outlier_indices


def BP_feature_extraction(df):
    print('-------------------------------------------------------------------------------------------')
    print('Starting BP feature extraction...')

    time = df['seconds']  # 访问'second'列
    fs = time.shape[0] / (time.iloc[-1] - time.iloc[0])  # 计算采样率
    print(f"Calculated Sampling Rate: {fs:.2f} Hz")
    s0 = df['s0_raw']  # 访问's0_raw'列
    s1 = df['s1_raw']  # 访问's1_raw'列
    SBP = df['SBP'].iloc[0]  # 访问'SBP'列的第一个值
    DBP = df['DBP'].iloc[0]  # 访问'DBP'列的第一个值


    # process every 20s data with 10s overlap
    data_length = len(df)
    window_size = int(20 * fs)  # 10s window
    overlap_size = int(15 * fs)  # 5s overlap
    start_indices = list(range(0, data_length - window_size + 1, window_size - overlap_size))

    HR = []
    PTT = []
    PWV = []
    SBP_groundtruth = [SBP] * len(start_indices)
    DBP_groundtruth = [DBP] * len(start_indices)


    for start_idx in start_indices:
        end_idx = start_idx + window_size
        print(f"\nProcessing data from index {start_idx} to {end_idx} (seconds {time.iloc[start_idx]:.2f} to {time.iloc[end_idx-1]:.2f})")
        s0_segment = s0[start_idx:end_idx].reset_index(drop=True)
        s1_segment = s1[start_idx:end_idx].reset_index(drop=True)
        seconds_segment = time[start_idx:end_idx].reset_index(drop=True)

        # Use the segment for further processing
        s0_raw = s0_segment
        s1_raw = s1_segment
        seconds = seconds_segment
        # signal processing ==========================================================================================
        print("-------------------------------------------------------------------------------------------")
        print("Starting signal processing...")

        # remove linear trend - detrend
        s0_x = np.arange(len(s0_raw))
        s0_p = np.polyfit(s0_x, s0_raw, 1)  # linear fit
        s0_trend = np.polyval(s0_p, s0_x)
        print('power coefficients for sensor 0 linear trend:', np.linalg.norm(s0_trend) / len(s0_trend))
        s0_detrended = s0_raw - s0_trend

        s1_x = np.arange(len(s1_raw))
        s1_p = np.polyfit(s1_x, s1_raw, 1)  # linear fit
        s1_trend = np.polyval(s1_p, s1_x)
        print('power coefficients for sensor 1 linear trend:', np.linalg.norm(s1_trend) / len(s1_trend))
        s1_detrended = s1_raw - s1_trend

        # bandpass filter - remove noise
        s0_filtered = butter_bandpass_filter(s0_detrended, 0.4, 8, fs)
        s1_filtered = butter_bandpass_filter(s1_detrended, 0.4, 8, fs)

        # hampel filter - remove outliers
        s0_filtered, s0_outliers = hampel_filter(s0_filtered, window_size=7, n_sigma=3)
        s1_filtered, s1_outliers = hampel_filter(s1_filtered, window_size=7, n_sigma=3)

        print(f"Sensor 0: Detected {len(s0_outliers)} outliers.")
        print(f"Sensor 1: Detected {len(s1_outliers)} outliers.")

        # Z-score normalization
        s0_norm = (s0_filtered - np.mean(s0_filtered)) / np.std(s0_filtered)
        s1_norm = (s1_filtered - np.mean(s1_filtered)) / np.std(s1_filtered)
        print('s0_norm mean:', np.mean(s0_norm))
        print('s1_norm mean:', np.mean(s1_norm))


        # systolic peak detection ===================================================================================
        print('-------------------------------------------------------------------------------------------')
        print('Systolic peak detection...')
        normal_HR_freq = [0.8, 2] / fs * len(s0_norm)  # normal human heart rate frequency range

        s0_fft = np.fft.rfft(s0_norm)
        s1_fft = np.fft.rfft(s1_norm)
        s0_fft_magnitude = np.abs(s0_fft) / max(np.abs(s0_fft))
        s1_fft_magnitude = np.abs(s1_fft) / max(np.abs(s1_fft))
        s0_fft_freq = np.fft.rfftfreq(len(s0_norm), d=1/fs)
        s1_fft_freq = np.fft.rfftfreq(len(s1_norm), d=1/fs)

        # detect HR from frequency peak
        s0_hr_indices = np.argmax(s0_fft_magnitude[round(normal_HR_freq[0]):round(normal_HR_freq[1])])
        s1_hr_indices = np.argmax(s1_fft_magnitude[round(normal_HR_freq[0]):round(normal_HR_freq[1])])
        s0_hr_indices += round(normal_HR_freq[0])  # adjust index
        s1_hr_indices += round(normal_HR_freq[0])  # adjust index
        s0_HR_freq = s0_fft_freq[s0_hr_indices] * 60  # convert to BPM
        s1_HR_freq = s1_fft_freq[s1_hr_indices] * 60  # convert to BPM
        print('HR estimated by sensor 0:', s0_HR_freq)
        print('HR estimated by sensor 1:', s1_HR_freq)
        HR_freq = (s0_HR_freq + s1_HR_freq) / 2

        # find systolic peaks
        peak_distance = int(fs * 60 / HR_freq * 0.8)  # minimum distance between peaks
        print(f"Calculated peak distance (in samples): {peak_distance}")
        s0_peaks, _ = find_peaks(s0_norm, distance=peak_distance)
        s1_peaks, _ = find_peaks(s1_norm, distance=peak_distance)
        print(f"Sensor 0: Detected {len(s0_peaks)} systolic peaks.")
        print(f"Sensor 1: Detected {len(s1_peaks)} systolic peaks.")

        # calculate HR from detected systolic peaks
        s0_HR_temp = (len(s0_peaks) - 3) / (seconds.iloc[s0_peaks[-2]] - seconds.iloc[s0_peaks[1]]) * 60
        s1_HR_temp = (len(s1_peaks) - 3) / (seconds.iloc[s1_peaks[-2]] - seconds.iloc[s1_peaks[1]]) * 60
        print('Sensor 0: HR calculated from detected peaks:', s0_HR_temp)
        print('Sensor 1: HR calculated from detected peaks:', s1_HR_temp)

        # average HR from temporal and frequency methods
        if HR:
            # the HR will not change much, so we use a weighted average
            HR.append(HR_freq * 0.3 + (s0_HR_temp + s1_HR_temp) / 2 * 0.3 + np.mean(HR) * 0.4)
        else:
            HR.append(HR_freq * 0.5 + (s0_HR_temp + s1_HR_temp) / 2 * 0.5)

        # onset detection ============================================================================================
        # TBD: implement onset detection if needed, can be the foot before each systolic peak

        # pulse transit time (PTT) calculation =======================================================================
        PTT_list = []
        max_PTT = 60 / HR[-1] * 0.3  # in seconds
        for i in range(min(len(s0_peaks), len(s1_peaks))):
            s0_peak_time = seconds.iloc[s0_peaks[i]]
            valid_s1_peaks = [p for p in s1_peaks if abs(seconds.iloc[p] - s0_peak_time) < max_PTT]
            if valid_s1_peaks:
                closest_t1 = min(valid_s1_peaks, key=lambda t1: abs(seconds.iloc[t1] - s0_peak_time))
                PTT_list.append(abs(seconds.iloc[closest_t1] - s0_peak_time))

        print('Average PTT (s):', np.mean(PTT_list), ', length of PTT list:', len(PTT_list))
        # pulse wave velocity (PWV) calculation =====================================================================
        PTT.append(np.mean(PTT_list))
        epsilon = 1e-6  # to avoid division by zero
        PWV.append(min(0.03 / (PTT[-1] + epsilon), 6))  # assuming distance between sensors is 0.03m

    return PTT, HR, PWV, SBP_groundtruth, DBP_groundtruth

# curve fitting for BP estimation ===========================================================================
def linear_regression_BP_estimation(PTT, HR, PWV, SBP_groundtruth, DBP_groundtruth):
    print('-------------------------------------------------------------------------------------------')
    print('Starting linear regression for BP estimation...')
    SBP_X = np.column_stack((PTT, HR))
    SBP_y = np.array(SBP_groundtruth)
    model = LinearRegression()
    model.fit(SBP_X, SBP_y)
    SBP_alpha, SBP_beta = model.coef_
    SBP_intercept = model.intercept_


    DBP_X = np.column_stack((PTT, HR))
    DBP_y = np.array(DBP_groundtruth)
    model = LinearRegression()
    model.fit(DBP_X, DBP_y)
    DBP_alpha, DBP_beta = model.coef_
    DBP_intercept = model.intercept_

    PWV2 = PWV**2
    PWV2 = PWV2.reshape(-1, 1)
    model = LinearRegression()
    model.fit(PWV2, SBP_y)
    SBP_a = model.coef_[0]
    SBP_b = model.intercept_

    model = LinearRegression()
    model.fit(PWV2, DBP_y)
    DBP_a = model.coef_[0]
    DBP_b = model.intercept_


    return SBP_alpha, SBP_beta, SBP_intercept, DBP_alpha, DBP_beta, DBP_intercept, SBP_a, SBP_b, DBP_a, DBP_b


# data loading ===============================================================================================
# laoding all csv files in the directory
file_list = glob.glob('ppg_data/*.csv')
PTT_all = []
HR_all = []
PWV_all = []
SBP_groundtruth_all = []
DBP_groundtruth_all = []
for file in file_list:
    print('-------------------------------------------------------------------------------------------')
    print(f"\nLoading data from file: {file}")
    df = pd.read_csv(file)
    df = df.dropna().reset_index(drop=True)  # drop NaN values and reset index
    PTT, HR, PWV, SBP_groundtruth, DBP_groundtruth = BP_feature_extraction(df)
    PTT_all.extend(PTT)
    HR_all.extend(HR)
    PWV_all.extend(PWV)
    SBP_groundtruth_all.extend(SBP_groundtruth)
    DBP_groundtruth_all.extend(DBP_groundtruth)

PTT_all = np.array(PTT_all)
HR_all = np.array(HR_all)
PWV_all = np.array(PWV_all)
SBP_groundtruth_all = np.array(SBP_groundtruth_all)
DBP_groundtruth_all = np.array(DBP_groundtruth_all)
# linear regression for BP estimation ===========================================================================
(
    SBP_alpha, SBP_beta, SBP_intercept, 
    DBP_alpha, DBP_beta, DBP_intercept, 
    SBP_a, SBP_b, DBP_a, DBP_b
) = linear_regression_BP_estimation(
     PTT_all, HR_all, PWV_all, SBP_groundtruth_all, DBP_groundtruth_all
)

# show model performance =======================================================================================
print('-------------------------------------------------------------------------------------------')
print('Linear Regression Results:')
print(f"Method 1 (SBP = {SBP_alpha:.4f} * PTT + {SBP_beta:.4f} * HR + {SBP_intercept:.2f})")
print(f"Method 1 (DBP = {DBP_alpha:.4f} * PTT + {DBP_beta:.4f} * HR + {DBP_intercept:.2f})")
print(f"Method 2 (SBP = {SBP_a:.4f} * PWV^2 + {SBP_b:.2f})")
print(f"Method 2 (DBP = {DBP_a:.4f} * PWV^2 + {DBP_b:.2f})")

SBP_pred_method1 = SBP_alpha * PTT_all + SBP_beta * HR_all + SBP_intercept
DBP_pred_method1 = DBP_alpha * PTT_all + DBP_beta * HR_all + DBP_intercept

PWV2 = PWV_all**2
SBP_pred_method2 = SBP_a * PWV2 + SBP_b
DBP_pred_method2 = DBP_a * PWV2 + DBP_b

SBP_MAE_method1 = np.mean(np.abs(SBP_groundtruth_all - SBP_pred_method1))
DBP_MAE_method1 = np.mean(np.abs(DBP_groundtruth_all - DBP_pred_method1))
SBP_MAE_method2 = np.mean(np.abs(SBP_groundtruth_all - SBP_pred_method2))
DBP_MAE_method2 = np.mean(np.abs(DBP_groundtruth_all - DBP_pred_method2))

SBP_STD_abs_method1 = np.std(np.abs(SBP_groundtruth_all - SBP_pred_method1))
DBP_STD_abs_method1 = np.std(np.abs(DBP_groundtruth_all - DBP_pred_method1))
SBP_STD_abs_method2 = np.std(np.abs(SBP_groundtruth_all - SBP_pred_method2))
DBP_STD_abs_method2 = np.std(np.abs(DBP_groundtruth_all - DBP_pred_method2))

print(f"SBP MAE + STD (Method 1): {SBP_MAE_method1:.2f} + {SBP_STD_abs_method1:.2f}")
print(f"DBP MAE + STD (Method 1): {DBP_MAE_method1:.2f} + {DBP_STD_abs_method1:.2f}")
print(f"SBP MAE + STD (Method 2): {SBP_MAE_method2:.2f} + {SBP_STD_abs_method2:.2f}")
print(f"DBP MAE + STD (Method 2): {DBP_MAE_method2:.2f} + {DBP_STD_abs_method2:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
axs[0].scatter(SBP_groundtruth_all, SBP_pred_method1, color='blue', label='Method1 Predictions', alpha=0.6)
axs[0].set_title('SBP Groundtruth vs. Predictions')
axs[0].set_xlabel('Groundtruth SBP')
axs[0].set_ylabel('Predicted SBP')
axs[0].legend()
axs[0].grid(True)
min_val = min(np.min(SBP_groundtruth_all), np.min(SBP_pred_method1))
max_val = max(np.max(SBP_groundtruth_all), np.max(SBP_pred_method1))
axs[0].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction', alpha=0.6)

axs[1].scatter(DBP_groundtruth_all, DBP_pred_method1, color='red', label='Method1 Predictions', alpha=0.6)
axs[1].set_title('DBP Groundtruth vs. Predictions')
axs[1].set_xlabel('Groundtruth DBP')
axs[1].set_ylabel('Predicted DBP')
axs[1].legend()
axs[1].grid(True)
min_val = min(np.min(DBP_groundtruth_all), np.min(DBP_pred_method1))
max_val = max(np.max(DBP_groundtruth_all), np.max(DBP_pred_method1))
axs[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction', alpha=0.6)


plt.tight_layout()
plt.show()










    


# # plot figures ==================================================================================================

# sns.set_style("whitegrid")
# sns.set_context("notebook", font_scale=1.1)  # 字体缩放
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))  # 2 rows, 1 column
# showpoint_num = seconds.shape[0] // 8 # number of points to show in time domain plot
# s0_peaks = s0_peaks[s0_peaks < showpoint_num]
# s1_peaks = s1_peaks[s1_peaks < showpoint_num]

# axs[0].plot(seconds[:showpoint_num], s0_norm[:showpoint_num], color="green", label="sensor 0:Filtered Data")  # scatter plot
# axs[0].plot(seconds[:showpoint_num], s1_norm[:showpoint_num], color="orange", label="sensor 1:Filtered Data")  # scatter plot
# axs[0].plot(seconds[s0_peaks], s0_norm[s0_peaks], "o", label="sensor 0:Detected Peaks", color="red")
# axs[0].plot(seconds[s1_peaks], s1_norm[s1_peaks], "o", label="sensor 1:Detected Peaks", color="blue")
# axs[0].set_title("PPG signal versus time", fontsize=14)
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("PPG Signal")
# axs[0].legend()
# axs[0].grid(True)

# showpoint_freq = round(10 / fs * len(s0_filtered))  # number of points to show in frequency domain plot
# axs[1].plot(s0_fft_freq[:showpoint_freq], s0_fft_magnitude[:showpoint_freq], label="sensor 0:FFT", color="green")
# axs[1].plot(s1_fft_freq[:showpoint_freq], s1_fft_magnitude[:showpoint_freq], label="sensor 1:FFT", color="orange")
# axs[1].set_title("PPG signal FFT", fontsize=14)
# axs[1].set_xlabel("Frequency (Hz)")
# axs[1].set_ylabel("Magnitude")
# axs[1].legend()
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()