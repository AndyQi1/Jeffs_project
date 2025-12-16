# 函数库
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

def remove_linear_trend(data):
    """
    移除数据的线性趋势（趋势线）
    参数：
        data: 输入数据（1D numpy数组或列表）
    返回：
        detrended_data: 去趋势后的数据
    """
    # 输入数据转为numpy数组
    data = np.array(data, dtype=float)
    x = np.arange(len(data))
    p = np.polyfit(x, data, 1)  # 线性拟合
    trend = np.polyval(p, x)
    detrended_data = data - trend
    return detrended_data

def PPG_feature_extraction(seconds, s0_raw, s1_raw):

    fs = seconds.shape[0] / (seconds[-1] - seconds[0])  # 计算采样率

    # signal processing ==========================================================================================

    # remove linear trend - detrend
    s0_detrended = remove_linear_trend(s0_raw)
    s1_detrended = remove_linear_trend(s1_raw)

    # bandpass filter - remove noise
    s0_filtered = butter_bandpass_filter(s0_detrended, 0.4, 8, fs)
    s1_filtered = butter_bandpass_filter(s1_detrended, 0.4, 8, fs)

    # hampel filter - remove outliers
    s0_filtered, s0_outliers = hampel_filter(s0_filtered, window_size=7, n_sigma=3)
    s1_filtered, s1_outliers = hampel_filter(s1_filtered, window_size=7, n_sigma=3)

    # Z-score normalization
    s0_norm = (s0_filtered - np.mean(s0_filtered)) / np.std(s0_filtered)
    s1_norm = (s1_filtered - np.mean(s1_filtered)) / np.std(s1_filtered)


    # systolic peak detection ===================================================================================
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
    HR_freq = (s0_HR_freq + s1_HR_freq) / 2

    # find systolic peaks
    peak_distance = int(fs * 60 / HR_freq * 0.8)  # minimum distance between peaks
    s0_peaks, _ = find_peaks(s0_norm, distance=peak_distance)
    s1_peaks, _ = find_peaks(s1_norm, distance=peak_distance)

    # calculate HR from detected systolic peaks
    s0_HR_temp = (len(s0_peaks) - 3) / (seconds[s0_peaks[-2]] - seconds[s0_peaks[1]]) * 60
    s1_HR_temp = (len(s1_peaks) - 3) / (seconds[s1_peaks[-2]] - seconds[s1_peaks[1]]) * 60

    # average HR from temporal and frequency methods
    HR = (HR_freq * 0.5 + (s0_HR_temp + s1_HR_temp) / 2 * 0.5)

    # onset detection ============================================================================================
    # TBD: implement onset detection if needed, can be the foot before each systolic peak

    # pulse transit time (PTT) calculation =======================================================================
    PTT_list = []
    max_PTT = 60 / HR * 0.3  # in seconds
    for i in range(min(len(s0_peaks), len(s1_peaks))):
        s0_peak_time = seconds[s0_peaks[i]]
        valid_s1_peaks = [p for p in s1_peaks if abs(seconds[p] - s0_peak_time) < max_PTT]
        if valid_s1_peaks:
            closest_t1 = min(valid_s1_peaks, key=lambda t1: abs(seconds[t1] - s0_peak_time))
            PTT_list.append(abs(seconds[closest_t1] - s0_peak_time))
    # pulse wave velocity (PWV) calculation =====================================================================
    PTT = (np.mean(PTT_list))
    epsilon = 1e-6  # to avoid division by zero
    PWV = min(0.03 / (PTT + epsilon), 6)  # assuming distance between sensors is 0.03m

    return PTT, HR, PWV, s0_norm, s1_norm

# curve fitting for BP estimation ===========================================================================
def BP_calculation(PTT, HR, PWV):
    SBP_alpha = 51.3264
    SBP_beta = -0.1230
    SBP_intercept = 118.00

    DBP_alpha = 148.4883    
    DBP_beta = -0.1954    
    DBP_intercept = 78.38

    SBP = SBP_alpha * PTT + SBP_beta * HR + SBP_intercept
    DBP = DBP_alpha * PTT + DBP_beta * HR + DBP_intercept

    return SBP, DBP
