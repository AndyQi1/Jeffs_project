# 函数库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, find_peaks, argrelmin
from scipy.stats import skew, pearsonr
# from scipy.integrate import trapz
from scipy.interpolate import interp1d
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import biobss
import neurokit2 as nk

# 函数处理算法：
# 1. 检测信号是否基本平坦（变化小于阈值）
# 2. 移除基线
# 3. 计算信号质量索引（SQI），同时预处理+检测特征点


def is_signal_flatlined(signal, fs, flat_time=0.5, flat_threshold=0.25, change_threshold=0.01):
    """
     判断信号是否基本平坦（变化小于阈值）
    参数：
        signal: 输入信号（1D numpy数组或列表）
        flat_time: 平坦时间（默认0.5秒）
        flat_threshold: 平坦阈值（默认0.25）
        change_threshold: 变化阈值（默认0.01）
    返回：
        True: 信号基本平坦
        False: 信号变化显著
    """
    flat_length = fs * flat_time
    flatline_segments = biobss.sqatools.detect_flatline_segments(
        signal, change_threshold=change_threshold, min_duration=flat_length
    )
    total_flatline_in_signal = np.sum([end - start for start, end in flatline_segments])
    # 检查最大变化率是否小于阈值
    return total_flatline_in_signal / len(signal) > flat_threshold


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

def calculate_onsets_from_peaks(ppg_filtered, peaks):
    """
    手动计算 PPG Onsets (波谷)
    逻辑：在两个波峰之间寻找最小值的索引
    """
    onsets = []
    # 确保波峰是排序的
    peaks = np.sort(peaks)
    
    # 转为 numpy array 以便处理
    ppg_signal = np.array(ppg_filtered)
    
    for i in range(len(peaks)):
        current_peak_idx = peaks[i]
        
        # 确定搜索范围
        # 如果是第一个峰，从信号开头搜到当前峰
        if i == 0:
            search_start = 0
            # 优化：为了防止开头太长，只往前搜平均心跳间隔的一半（比如0.5秒）
            # search_start = max(0, current_peak_idx - 50) 
        else:
            search_start = peaks[i-1]
            
        search_end = current_peak_idx
        
        # 如果两个峰挨得太近（比如误检），跳过
        if search_end - search_start < 5:
            continue
            
        # 截取片段
        segment = ppg_signal[search_start:search_end]
        
        # 找最小值的相对索引
        if len(segment) > 0:
            min_idx_local = np.argmin(segment)
            min_idx_global = search_start + min_idx_local
            onsets.append(min_idx_global)
    
    return np.array(onsets)

def detect_dicrotic_notch(ppg_filtered, peak_idx, next_onset_idx, fs):
    """
    检测重播切迹
    参数：
        ppg_signal: PPG信号
        peak_idx: 收缩峰索引
        onset_idx: 起点索引
        fs: 采样频率
    返回：
        notch_idx: 重播切迹索引
    """
    # 1. 确定搜索范围：收缩峰到下一个起点之间
    search_start = peak_idx
    search_end = next_onset_idx  # 限制在400ms内
    
    # 2. 在搜索范围内寻找局部最小值
    search_segment = ppg_filtered[search_start:search_end]
    
    if len(search_segment) < 10:  # 太短的片段无法检测
        return None
    
    local_min_indices = argrelmin(search_segment)[0]
    
    if len(local_min_indices) == 0:
        # print("No local minimum found in search range")
        return None
    
    local_min_idx = local_min_indices[0]
    # 转换为全局索引
    global_min_idx = search_start + local_min_idx
    
    # 计算切迹深度（相对于收缩峰）
    notch_depth = ppg_filtered[peak_idx] - ppg_filtered[global_min_idx]
    
    # 计算信号幅度（收缩峰到下一个起点的差值）
    signal_amplitude = ppg_filtered[peak_idx] - ppg_filtered[next_onset_idx]

    best_notch_idx = global_min_idx
        
    
    return best_notch_idx

def calculate_pi(ppg_raw, peaks, onsets):
    pi_values = []

    # 确保每个峰值都有对应的起点（对齐）
    # 通常一个脉搏波由一个 onset 和随后的一个 peak 组成
    for i in range(len(onsets)):
        # 寻找该 onset 之后的最近一个 peak
        current_onset_idx = onsets[i]
        future_peaks = peaks[peaks > current_onset_idx]

        if len(future_peaks) > 0:
            current_peak_idx = future_peaks[0]

            # AC 分量 = 峰值 - 谷值
            # DC 分量 = 谷值（代表非搏动性血液/组织吸收）
            ac = ppg_raw[current_peak_idx] - ppg_raw[current_onset_idx]

            dc = ppg_raw[current_onset_idx]

            if dc != 0:
                pi = (ac / dc) * 100
                pi_values.append(pi)
    if len(pi_values) == 0:
        return 0.0

    return np.mean(pi_values)  # 返回平均灌注指数

def extract_pulse_waveform(ppg_filtered, peaks, onsets, fs):
    """
    从PPG信号中提取有效波形
    参数：
        ppg_filtered: 输入PPG信号（1D numpy数组或列表）
        peaks: 峰值索引列表
        onsets: 谷值索引列表
        fs: 采样频率（Hz）
    返回：
        valid_waveforms: 有效波形列表（每个元素为1D numpy数组）
    """
    # 存储分段后的波形


    pulse_waveforms = []

    # 循环：使用 onsets[i] 到 onsets[i+1] 作为一个完整周期
    for i in range(len(onsets) - 1):
        start_idx = onsets[i]
        end_idx = onsets[i + 1]

        # 提取这一段波形（建议使用滤波后的信号）
        pulse = ppg_filtered[start_idx:end_idx]

        # --- 边界过滤逻辑 ---

        # 1. 检查这段波形中间是否包含且仅包含一个 Peak
        # 这是确保波形完整（不是半个波形）的核心逻辑
        peaks_in_segment = peaks[(peaks > start_idx) & (peaks < end_idx)]

        if len(peaks_in_segment) == 1 and len(pulse) > fs*0.4 and len(pulse) < fs*2:
            # 说明这是一个完美的 [谷 -> 峰 -> 谷] 结构
            notch_idx = detect_dicrotic_notch(ppg_filtered, peaks_in_segment[0], end_idx, fs)
            if notch_idx is None:
                notch_idx = int(0.5 * (peaks_in_segment[0] + start_idx))
                
            pulse_waveforms.append(
                {
                    "signal": pulse,
                    "onset_idx": start_idx,
                    "notch_idx": notch_idx,
                    "peak_idx": peaks_in_segment[0],
                    "length": len(pulse),
                }
            )
        else:
            # 如果这一段里没有峰值，或者有多个峰值，说明检测失误，丢弃
            continue

    return pulse_waveforms

def calculate_similarity(pulse_waveforms, target_len=100):
    """
    输入:
    pulse_waveforms: 输入脉冲波列表（每个元素为字典，包含"signal"键）
    fs: 采样频率（Hz）
    target_len: 重采样长度，用于对齐波形
    """
    if len(pulse_waveforms) < 3:
        return 0.0
    pulses = []
    # --- 第一步：提取并归一化每个脉冲 ---
    for i in range(len(pulse_waveforms)):
            
        segment = pulse_waveforms[i]["signal"]

        # 重采样到统一长度 (对齐)
        x_old = np.linspace(0, 1, len(segment))
        f = interp1d(x_old, segment, kind='cubic')
        resampled = f(np.linspace(0, 1, target_len))
        
        # Z-score 归一化 (消除振幅影响)
        normalized = (resampled - np.mean(resampled)) / np.std(resampled)
        pulses.append(normalized)

    # --- 第二步：生成模板 (Template) ---
    # 使用中位数合成模板，比平均值更抗噪
    template = np.median(pulses, axis=0)

    # --- 第三步：计算每个波形与模板的相关性 ---
    correlations = []
    for p in pulses:
        corr, _ = pearsonr(p, template)
        correlations.append(corr)

    # --- 第四步：计算最终得分 ---
    # 结果是所有脉搏波相似度的平均值
    similarity_score = np.mean(correlations)
    
    return similarity_score

def extract_ppg_waveforms_with_quality_index(ppg_detrend, ppg_remove_outliers, fs):
    """
    计算信号质量索引（SQI）
    参数：
        signal: 输入信号（1D numpy数组或列表）
        fs: 采样频率（Hz）
    返回：
        results: 包含信号质量索引和其他相关信息的字典
    """
    results = {}
    # 计算信号是否flatline
    results["is_flatlined"] = is_signal_flatlined(ppg_remove_outliers, fs)  # 检查原始信号是否flatline
    if results["is_flatlined"]:
        return results
    
    signals, info = nk.ppg_process(ppg_detrend, sampling_rate=fs)
    ppg_filtered = signals["PPG_Clean"].values
    if 'PPG_Onsets' not in info or len(info['PPG_Onsets']) == 0:
        # print("手动计算 Onsets...") # 调试用
        peaks = info["PPG_Peaks"]
        
        # 调用上面的函数
        calculated_onsets = calculate_onsets_from_peaks(ppg_filtered, peaks)
        
        # 强制写入 info 字典，解决后续 KeyError
        info['PPG_Onsets'] = calculated_onsets

    peaks = info["PPG_Peaks"]
    onsets = info["PPG_Onsets"]
    # 计算perfusion index（基于原始信号）
    results["perfusion_index"] = calculate_pi(ppg_remove_outliers, peaks, onsets)
    # 计算skewness（基于滤波后的信号）
    results["skewness"] = skew(ppg_filtered)
    # 计算脉搏波相似度（基于滤波后z正则化之后的信号）
    pulse_waveforms = extract_pulse_waveform(ppg_filtered, peaks, onsets, fs)
    results["pulse_waveforms"] = pulse_waveforms
    results["similarity"] = calculate_similarity(pulse_waveforms)
    # 计算SVRI（基于滤波后的信号）
    SVRI = []
    for pulse_waveform in pulse_waveforms:
            pulse = pulse_waveform["signal"]
            peak_idx = pulse_waveform["peak_idx"]
            start_idx = pulse_waveform["onset_idx"]
            pulse_scaled = _scale(pulse)
            SVRI.append(np.mean(pulse_scaled[peak_idx - start_idx:]) / np.mean(pulse_scaled[:peak_idx - start_idx]))
    results["SVRI"] = np.mean(SVRI)
    # 计算IPA
    IPA = []
    for pulse_waveform in pulse_waveforms:
            pulse = pulse_waveform["signal"]
            start_idx = pulse_waveform["onset_idx"]
            if pulse_waveform["notch_idx"]:
                notch_idx = pulse_waveform["notch_idx"]
            else:
                continue
            pulse_normalized = abs((pulse - np.mean(pulse)) / np.std(pulse)) # 区域比例应该是绝对值？？？
            sys_phase = pulse_normalized[:notch_idx - start_idx]
            dia_phase = pulse_normalized[notch_idx - start_idx:]
            sys_x = np.linspace(0, len(sys_phase) - 1, len(sys_phase))
            dia_x = np.linspace(0, len(dia_phase) - 1, len(dia_phase))
            sys_value = np.trapezoid(sys_phase, sys_x)
            dia_value = np.trapezoid(dia_phase, dia_x)
            IPA.append(sys_value / dia_value)
    results["IPA"] = np.mean(IPA)
    
    results["peak_idx"] = peaks
    results["onset_idx"] = onsets

    results["ppg_signal"] = ppg_filtered


    return results

def calculate_hr(ppg_signal, fs):
    """
    计算PPG信号的心率（HR）
    参数：
        ppg_signal: 输入PPG信号（1D numpy数组或列表）
        fs: 采样频率（Hz）
    返回：
        hr: 心率（BPM）
    """
    # 提取有效波形
    if len(ppg_signal) < 2 * fs:
        return None
    window = np.hanning(len(ppg_signal))
    ppg_windowed = ppg_signal * window
    n_fft = int(fs * 60)  
    spectrum = np.abs(np.fft.rfft(ppg_windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)

    # 4.1 限制在人心率范围内 [40 BPM ~ 220 BPM]
    valid_mask = (freqs >= 40/60) & (freqs <= 220/60)
    valid_freqs = freqs[valid_mask]
    valid_spectrum = spectrum[valid_mask]
    
    # 如果信号全被滤没了（极端情况）
    if len(valid_spectrum) == 0:
        return None

    # 4.2 谐波增强 (Harmonic Summation) - 关键步骤
    # 很多人测不准是因为把“心率的倍频”当成了心率
    # 我们构造一个增强谱：Enhanced(f) = Base(f) + 0.5 * Base(2f)
    
    enhanced_spectrum = np.copy(valid_spectrum)
    
    for i, f in enumerate(valid_freqs):
        # 寻找当前频率 f 的 2倍频 (2f) 在哪里
        target_f = f * 2
        # 在 valid_freqs 里找最接近 target_f 的索引
        if target_f <= valid_freqs[-1]:
            idx = np.argmin(np.abs(valid_freqs - target_f))
            # 将倍频的能量折半加回基频
            enhanced_spectrum[i] += 0.5 * valid_spectrum[idx]

    # 5. 寻找最大峰值 (Peak Finding)
    best_idx = np.argmax(enhanced_spectrum)
    best_freq = valid_freqs[best_idx]
    estimated_bpm = best_freq * 60
        
    return estimated_bpm

def extract_valid_waveforms(ppg0_signal, ppg1_signal, fs, segment_length=10, min_SQI=0.7):
    """
    从PPG信号中提取有效波形
    参数：
        ppg0_signal: 输入PPG信号（1D numpy数组或列表）
        ppg1_signal: 输入PPG信号（1D numpy数组或列表）
        fs: 采样频率（Hz）
        min_SQI: 最小信号质量索引（默认0.7）
    返回：
        valid_waveforms_0: 有效波形列表（每个元素为1D numpy数组）
        valid_waveforms_1: 有效波形列表（每个元素为1D numpy数组）
    """

    # 先对整体信号去除基线漂移和滤波（去除离散点->去趋势->滤波）
    ppg0_remove_outliers, _ = hampel_filter(ppg0_signal, window_size=5, n_sigma=3)
    ppg0_detrend = remove_linear_trend(ppg0_remove_outliers)

    ppg1_remove_outliers, _ = hampel_filter(ppg1_signal, window_size=5, n_sigma=3)
    ppg1_detrend = remove_linear_trend(ppg1_remove_outliers)


    # 再对预处理后的信号进行分段提取有效波形
    valid_waveforms_0 = []
    valid_waveforms_1 = []
    
    window_size = int(segment_length * fs)
    overlap_size = int(window_size // 2)
    step_size = window_size - overlap_size
    for start in range(0, len(ppg0_detrend) - window_size + 1, step_size):
        segment_ppg0_detrend = ppg0_detrend[start : start + window_size]
        segment_ppg1_detrend = ppg1_detrend[start : start + window_size]
        segment_ppg0_remove_outliers = ppg0_remove_outliers[start : start + window_size]
        segment_ppg1_remove_outliers = ppg1_remove_outliers[start : start + window_size]

        results_0 = extract_ppg_waveforms_with_quality_index(segment_ppg0_detrend, segment_ppg0_remove_outliers, fs)
        results_1 = extract_ppg_waveforms_with_quality_index(segment_ppg1_detrend, segment_ppg1_remove_outliers, fs)
        if results_0["is_flatlined"] or results_1["is_flatlined"]:
            continue
        else:
            if ((results_0['similarity'] < 0.5 and results_1['similarity'] < 0.5) or 
                (results_0['skewness'] < 0 and  results_1['skewness'] < 0)):
                continue
            results_0["start_idx"] = start
            results_1["start_idx"] = start
            # # 映射到0到1之间
            # SQI_sim_0 = np.clip((results_0['similarity'] - 0.6) / (0.9 - 0.6), 0, 1)
            # SQI_sim_1 = np.clip((results_1['similarity'] - 0.6) / (0.9 - 0.6), 0, 1)
            # if 0.8 <= results_0['skewness'] <= 2.0:
            #     SQI_skew_0 = 1.0        
            # elif 0 < results_0['skewness'] < 0.8:
            #     SQI_skew_0 = results_0['skewness'] / 0.8  # 线性上升
            # else: # skew > 2.0
            #     SQI_skew_0 = max(0, 1.0 - (results_0['skewness'] - 2.0) * 0.5) # 缓慢下降
            # SQI_pi_0 = np.clip((results_0['perfusion_index'] - 0.1) / (1.0 - 0.1), 0, 1)
            # if 0.8 <= results_1['skewness'] <= 2.0:
            #     SQI_skew_1 = 1.0        
            # elif 0 < results_1['skewness'] < 0.8:
            #     SQI_skew_1 = results_1['skewness'] / 0.8  # 线性上升
            # else: # skew > 2.0
            #     SQI_skew_1 = max(0, 1.0 - (results_1['skewness'] - 2.0) * 0.5) # 缓慢下降
            # SQI_pi_1 = np.clip((results_1['perfusion_index'] - 0.1) / (1.0 - 0.1), 0, 1)
            # w_sim, w_skew, w_pi = 0.5, 0.3, 0.2
            # SQI_0 = w_sim * SQI_sim_0 + w_skew * SQI_skew_0 + w_pi * SQI_pi_0
            # SQI_1 = w_sim * SQI_sim_1 + w_skew * SQI_skew_1 + w_pi * SQI_pi_1
            # results_0["SQI"] = SQI_0
            # results_1["SQI"] = SQI_1
            # SQI = 0.5 * (SQI_0 + SQI_1)
            # if SQI >= min_SQI:
            
            valid_waveforms_0.append(results_0)
            valid_waveforms_1.append(results_1)
        
    return valid_waveforms_0, valid_waveforms_1

def _scale(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_scaled = (data - data_min) / (data_max - data_min)
    return data_scaled
def extract_ppg_features(s0_raw, s1_raw, segment_length=10, fs=500):
    """
    从PPG信号中提取特征
    参数：
        s0_raw: 输入PPG信号（1D numpy数组或列表）
        s1_raw: 输入PPG信号（1D numpy数组或列表）
        segment_length: 分段长度（秒）
        fs: 采样频率（Hz）
    返回：
        PTT_list: 脉冲 transit time 列表（每个元素为浮点数）
        HR_list: 心率列表（每个元素为浮点数）
        PWV_list: 脉冲波速度列表（每个元素为浮点数）
        SVRI_list: 信号质量索引列表（每个元素为浮点数）
        IPA_list: 脉冲波面积列表（每个元素为浮点数）
        skew_list: 波形偏斜度列表（每个元素为浮点数）
        similarity_list: 波形相似度列表（每个元素为浮点数）
        perfusion_index_list:  perfusion index 列表（每个元素为浮点数）
        ppg0_waveforms: 有效PPG波形列表（每个元素为字典）
        ppg1_waveforms: 有效PPG波形列表（每个元素为字典）
    """
    if len(s0_raw) < 3 * fs or len(s1_raw) < 3 * fs:
        return None
    segment_length = min(len(s0_raw) // fs, len(s1_raw) // fs, segment_length)
    min_SQI = 0.7

    ppg0_waveforms, ppg1_waveforms = extract_valid_waveforms(s0_raw, s1_raw, fs, segment_length, min_SQI)
    HR_list = []
    PTT_list = []
    PWV_list = []
    SVRI_list = []
    IPA_list = []
    skew_list = []
    similarity_list = []
    perfusion_index_list = []
        
    max_time_diff = 0.2
    max_sample_diff = int(max_time_diff * fs)
    min_waveforms = min(len(ppg0_waveforms), len(ppg1_waveforms))
    if min_waveforms == 0:
        return [], [], [], [], [], [], [], [], ppg0_waveforms, ppg1_waveforms
    for i in range(min_waveforms):
        ppg0 = ppg0_waveforms[i]["ppg_signal"]
        ppg1 = ppg1_waveforms[i]["ppg_signal"]

        hr0 = calculate_hr(ppg0, fs)
        hr1 = calculate_hr(ppg1, fs)
        if hr0 is None or hr1 is None:
            continue
        else:
            HR_list.append(1/2 * (hr0 + hr1))

        # pulse transit time (PTT) and pulse wave velocity (PWV) calculation =======================================================================
        ppg0_peaks = ppg0_waveforms[i]["peak_idx"]
        ppg1_peaks = ppg1_waveforms[i]["peak_idx"]

        PTT = []
        PWV = []
        for j in range(len(ppg0_peaks)):
            ppg0_peak_idx = ppg0_peaks[j]
            if len(ppg1_peaks) == 0:
                continue
            closest_arg = np.abs(ppg1_peaks - ppg0_peak_idx).argmin()
            ppg1_peak_idx = ppg1_peaks[closest_arg]
            sample_diff = abs(ppg1_peak_idx - ppg0_peak_idx)
            if sample_diff < max_sample_diff and sample_diff > 0:
                time_diff = sample_diff / fs
                PTT.append(time_diff)
                if time_diff > 0:
                    PWV.append(0.02 / (PTT[-1]))
        PTT_list.append(np.mean(PTT))
        PWV_list.append(np.mean(PWV))
        # return SVRI =======================================================================
        
        SVRI_list.append(0.5 * (ppg0_waveforms[i]["SVRI"] + ppg1_waveforms[i]["SVRI"]))

        # return IPA =======================================================================
        IPA_list.append(0.5 * (ppg0_waveforms[i]["IPA"] + ppg1_waveforms[i]["IPA"]))

        # return SQI =======================================================================
        skew_list.append(0.5 * (ppg0_waveforms[i]["skewness"] + ppg1_waveforms[i]["skewness"]))
        similarity_list.append(0.5 * (ppg0_waveforms[i]["similarity"] + ppg1_waveforms[i]["similarity"]))
        perfusion_index_list.append(0.5 * (ppg0_waveforms[i]["perfusion_index"] + ppg1_waveforms[i]["perfusion_index"]))

    return PTT_list, HR_list, PWV_list, SVRI_list, IPA_list, skew_list, similarity_list, perfusion_index_list, ppg0_waveforms, ppg1_waveforms  


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


if __name__ == "__main__":
    import os
    filename = "ppg_data\ZYD_ppg_data_2025-10-28T08-59-27.363Z.csv"
    filepath = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(filepath)
    seconds = df["seconds"].values
    s0_raw = df["s0_raw"].values
    s1_raw = df["s1_raw"].values
    fs = int(seconds.shape[0] / (seconds[-1] - seconds[0]))  # 计算采样率
    PTT_list, HR_list, PWV_list, SVRI_list, IPA_list, skew_list, similarity_list, perfusion_index_list, ppg0_waveforms, ppg1_waveforms = extract_ppg_features(s0_raw, s1_raw, segment_length=10, fs=fs)
    print("PTT_list:", PTT_list)
    print("HR_list:", HR_list)
    print("PWV_list:", PWV_list)
    print("SVRI_list:", SVRI_list)
    print("IPA_list:", IPA_list)
    print("skew_list:", skew_list)
    print("similarity_list:", similarity_list)
    print("perfusion_index_list:", perfusion_index_list)
