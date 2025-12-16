from flask import Flask, request, render_template_string, make_response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import csv
from io import StringIO
from datetime import datetime
from function_lib import *
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# 存储最新的BPM值和原始数据
latest_bpm = {'s0': 0.0, 's1': 0.0}
latest_bp = {'sbp': 0.0, 'dbp': 0.0}
raw_data_history = []  # 服务器端历史（备用）
is_pushing = False     # 控制推送状态
start_time = None


@app.route('/')
def index():
    return render_template_string(r"""
    <html>
    <head>
        <title>PPG Sensor Monitor</title>
        <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
        <style>
            .control-panel { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; }
            button { padding: 10px 20px; margin-right: 10px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            #startBtn { background-color: #4CAF50; color: white; }
            #stopBtn { background-color: #f44336; color: white; display: none; }
            #saveBtn { background-color: #2196F3; color: white; display: inline-block; }
            .chart-container { width: 800px; height: 300px; margin: 40px 0; }                     
            .value-display { font-size: 18px; margin: 10px 0; padding: 10px; border: 1px solid #eee; border-radius: 5px; }
            .sensor-label { font-weight: bold; margin-right: 20px; }
            .timer { color: #666; margin-left: 20px; font-size: 14px; }
            h3 + .chart-container + h3 { 
                margin-top: 80px;
                margin-bottom: 10px; 
            }
            h3:not(:last-of-type) {
                margin-bottom: 10px;
            }
            #saveBtn {
                background-color: #2196F3; /* 正常可点击：蓝色 */
                color: white;
                display: inline-block; /* 一直显示，不隐藏 */
            }

            /* 禁用状态（初始/监控中）：灰色 */
            #saveBtn:disabled {
                background-color: #9E9E9E; /* 灰色背景 */
                color: #FFFFFF; /* 灰色文字 */
                cursor: not-allowed; /* 鼠标显示“禁止”图标 */
            }
            #estimateBtn:not(:disabled):not(.running) {
                background-color: #2196F3; /* 亮蓝色（醒目且与现有按钮区分） */
                color: white;
            }

            /* BP Estimation按钮 - 运行中状态（显示"Stop BP Estimation"） */
            #estimateBtn:not(:disabled).running {
                background-color: #FF9800; /* 橙色（与蓝色强对比，明确指示运行中） */
                color: white;
            }

            /* 禁用状态（监控未开始时，保持灰色但更清晰） */
            #estimateBtn:disabled {
                background-color: #9E9E9E; /* 中灰色（比原来稍深，明确禁用） */
                color: #FFFFFF;
                cursor: default;
            }
            
            /* 1. Flex父容器：横向排列，间距均匀 */
            .charts-row {
                display: flex;          /* 横向一行排列 */
                gap: 15px;              /* 图表之间的间距（避免拥挤） */
                width: 100%;            /* 占满页面宽度 */
                margin: 40px 0;         /* 上下整体间距 */
            }

            /* 2. RAW图表（正常大小）：宽度占比最大 */
            .chart-container {
                flex: 1.2;              /* 宽度权重最大（占~45%） */
                height: 260px;          /* 高度适中 */
                min-width: 280px;       /* 最小宽度，避免变形 */
            }

            /* 3. BPM图表（缩小）：宽度和高度都更小 */
            .bpm-chart-small {
                flex: 0.7;              /* 宽度权重小（占~27%） */
                height: 220px;          /* 高度比RAW矮40px */
            }

            /* 4. BP图表（缩小）：和BPM尺寸一致 */
            .bp-chart-small {
                flex: 0.7;              /* 宽度和BPM相同（占~27%） */
                height: 220px;          /* 高度和BPM相同 */
            }

            /* 5. 响应式适配：屏幕窄时（<1200px）自动纵向排列 */
            @media (max-width: 1200px) {
                .charts-row {
                    flex-direction: column;  /* 纵向排列 */
                }
                /* 窄屏时恢复原尺寸，避免挤乱 */
                .chart-container, .bpm-chart-small, .bp-chart-small {
                    flex: none;
                    width: 800px;
                    height: 300px;
                    margin: 20px 0;
                }
            }

            /* 6. 图表标题优化：居中+小字体，不占空间 */
            .chart-container h3 {
                text-align: center;
                font-size: 15px;
                margin-bottom: 8px;
                color: #333;
            }
        </style>       
    </head>
    <body>
        <h2>❤️ PPG Sensor Monitor</h2>
        
        <div class="control-panel">
            <button id="startBtn">Start Monitoring</button>
            <button id="stopBtn">Stop Monitoring</button>
            <button id="estimateBtn" disabled>Estimate BP</button>
            <button id="saveBtn" disabled>Save Data (CSV)</button>
            <span style="margin-left:15px;">True SBP：</span>
            <input type="number" id="sbpInput" placeholder="systolic BP" value="0" min="0" style="padding:8px; width:120px;">
            <span style="margin-left:10px;">True DBP：</span>
            <input type="number" id="dbpInput" placeholder="diastolic BP" value="0" min="0" style="padding:8px; width:120px;">                      
            <span id="status">Status: Not Started</span>
            <span class="timer" id="autoStopTimer">Countdown--:--</span>
        </div>

        <div class="value-display">
            <span class="sensor-label">Sensor 0:</span>
            HR: <span id="s0-bpm" style="color:rgb(54, 162, 235);">0</span> | 
            Last Raw: <span id="s0-raw" style="color:rgb(54, 162, 235);">0</span>
            <br>
            <span class="sensor-label">Sensor 1:</span>
            HR: <span id="s1-bpm" style="color:rgb(255, 99, 132);">0</span> | 
            Last Raw: <span id="s1-raw" style="color:rgb(255, 99, 132);">0</span>
                                              
            <span class="sensor-label">Estimated BP:</span>
            SBP: <span id="est-sbp" style="color:rgb(75, 192, 192);">--</span> mmHg | 
            DBP: <span id="est-dbp" style="color:rgb(75, 192, 192);">--</span> mmHg |
            <span id="est-status" style="color:rgb(128, 128, 128);">Click "Start BP Estimation" to calculate</span>
        </div>

        <div class="charts-row">
            <!-- 1. RAW数据图表 -->
            <div class="chart-container">
                <h3>Raw Data Chart</h3>
                <canvas id="raw-chart"></canvas>
            </div>

            <!-- 2. BPM图表（要更小，加专属类） -->
            <div class="chart-container bpm-chart-small">
                <h3>HR Chart</h3>
                <canvas id="bpm-chart"></canvas>
            </div>

            <!-- 3. BP图表 -->
            <div class="chart-container bp-chart-small">
                <h3>Real-Time BP Chart</h3>
                <canvas id="bp-chart"></canvas>
            </div>
        </div>

        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const socket = io();
                let monitoringActive = false;
                let startTimestamp = null;  // 基准时间（毫秒，开始监控时的系统时间）
                let rawDataHistory = [];    // 保存所有数据（用于CSV）
                let autoStopTimer = null;   // 自动停止定时器ID
                const AUTO_STOP_DURATION = 120000;  // 2分钟（毫秒）
                let countdownInterval = null;       // 倒计时定时器
                const MAX_RAW_POINTS = 500;         // 最多显示500个RAW点
                const MAX_BPM_POINTS = 10;        // 最多显示10个BPM点
                const MAX_BP_POINTS = 10;   // 血压图表最多显示10个点

                // DOM元素
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const saveBtn = document.getElementById('saveBtn');
                const statusElem = document.getElementById('status');
                const timerElem = document.getElementById('autoStopTimer');
                const sbpInput = document.getElementById('sbpInput');
                const dbpInput = document.getElementById('dbpInput'); 
                const estimateBtn = document.getElementById('estimateBtn');
                const estSbpElem = document.getElementById('est-sbp');
                const estDbpElem = document.getElementById('est-dbp');
                const estStatusElem = document.getElementById('est-status');
                                  
                let isBpEstimating = false;  // 是否处于BP持续测量模式
                let bpInterval = null;       // BP测量定时任务ID
                const BP_UPDATE_INTERVAL = 3000;  // BP更新间隔（3秒，毫秒）
                                 
                                  
                let globalLastTimestamp = null;  // 全局最后一个点的时间戳（确保递增）
                                  
                let sbpValue = 0;
                let dbpValue = 0;
                sbpInput.addEventListener('input', function() {
                    sbpValue = parseFloat(this.value) || 0;
                    // 发送到服务器
                    fetch('/bp', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                        body: `sbp=${sbpValue}`
                    });
                });
                dbpInput.addEventListener('input', function() {
                    dbpValue = parseFloat(this.value) || 0;
                    // 发送到服务器
                    fetch('/bp', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                        body: `dbp=${dbpValue}`
                    });
                });                  
                
                                  
                const THEORETICAL_SAMPLE_RATE = 500;  // 理论采样率：500Hz
                const BASE_INTERVAL = 1000 / THEORETICAL_SAMPLE_RATE;  // 理论间隔：2ms
                const ACTUAL_INTERVAL_FACTOR = 1.2;  // 实际间隔系数（留1.2倍弹性，适配采样率降低）
                const SAFE_INTERVAL = BASE_INTERVAL * ACTUAL_INTERVAL_FACTOR;  // 安全间隔：4.8ms（兼容实际波动）

                // 开始监控逻辑（核心：记录基准时间）
                startBtn.addEventListener('click', function() {
                    monitoringActive = true;
                    estimateBtn.disabled = false;
                    estimateBtn.textContent = 'Start BP Estimation';  // 初始文字              
                    startTimestamp = new Date().getTime();  // 基准时间（毫秒，如1620000000000）
                    rawDataHistory = [];
                    socket.emit('control', { action: 'start' });

                    // UI更新
                    startBtn.style.display = 'none';
                    stopBtn.style.display = 'inline-block';
                    saveBtn.disabled = true;
                    
                    statusElem.textContent = 'Status: Monitoring';
                    resetCharts();

                    // 设置2分钟自动停止
                    clearTimeout(autoStopTimer);
                    autoStopTimer = setTimeout(function() {
                        if (monitoringActive) {
                            statusElem.textContent = 'Status: Auto-stop (2 min reached)';
                            stopMonitoring();
                        }
                    }, AUTO_STOP_DURATION);

                    // 启动倒计时
                    startCountdown();
                });

                // 停止监控逻辑
                stopBtn.addEventListener('click', function() {
                    statusElem.textContent = 'Status: Manual Stop';
                    stopMonitoring();
                    
                });

                function stopMonitoring() {
                    monitoringActive = false;
                    socket.emit('control', { action: 'stop' });
                    // 新增：停止BP测量（若正在运行）
                    if (isBpEstimating) {
                        isBpEstimating = false;
                        clearInterval(bpInterval);
                        estimateBtn.textContent = 'Start BP Estimation';
                        estStatusElem.textContent = 'BP Estimation Stopped (Monitoring Ended)';
                        estStatusElem.style.color = 'orange';
                    }

                    // 清除定时器
                    clearTimeout(autoStopTimer);
                    clearInterval(countdownInterval);

                    // UI更新
                    stopBtn.style.display = 'none';
                    startBtn.style.display = 'inline-block';
                    saveBtn.disabled = false;
                    timerElem.textContent = 'Auto-stop Countdown: --:--';
                                  
                    estimateBtn.disabled = true;
                    estSbpElem.textContent = '--';
                    estDbpElem.textContent = '--';
                    estStatusElem.textContent = 'Click "Estimate BP" to calculate';    
                }
                estimateBtn.addEventListener('click', function() {
                    if (!monitoringActive) {
                        estStatusElem.textContent = 'Error: Start monitoring first!';
                        estStatusElem.style.color = 'red';
                        return;
                    }

                    // 状态1：未开始BP测量→启动
                    if (!isBpEstimating) {
                        isBpEstimating = true;
                        estimateBtn.textContent = 'Stop BP Estimation';  // 按钮文字改为停止
                        estimateBtn.classList.add('running');
                        estStatusElem.textContent = 'BP Estimation Running';
                        estStatusElem.style.color = 'green';
                        
                        // 立即计算一次，再启动5秒定时
                        autoCalculateBP();
                        bpInterval = setInterval(autoCalculateBP, BP_UPDATE_INTERVAL);
                    } 
                    // 状态2：已开始BP测量→停止
                    else {
                        isBpEstimating = false;
                        estimateBtn.textContent = 'Start BP Estimation';  // 按钮文字恢复
                        estimateBtn.classList.remove('running');
                        estStatusElem.textContent = 'BP Estimation Stopped';
                        estStatusElem.style.color = 'orange';
                        
                        // 清除定时任务
                        clearInterval(bpInterval);
                    }
                });
                async function autoCalculateBP() {
                    if (!monitoringActive || !isBpEstimating) return;

                    // 1. 筛选最新20秒数据（与原逻辑一致）
                    const latestTime = rawDataHistory[rawDataHistory.length - 1].seconds;
                    const windowStartTime = Math.max(0, latestTime - 20);
                    const recent20sData = rawDataHistory.filter(entry => 
                        entry.seconds >= windowStartTime && entry.seconds <= latestTime
                    );

                    // 2. 数据不足处理
                    if (recent20sData.length < 100) {
                        const msg = `Insufficient data (${recent20sData.length} points in 20s)`;
                        estStatusElem.textContent = `BP Estimation: ${msg}`;
                        estStatusElem.style.color = 'orange';
                        return;
                    }

                    // 3. 发送请求到后端（与原逻辑一致）
                    const requestData = {
                        seconds: recent20sData.map(entry => entry.seconds),
                        s0_raw: recent20sData.map(entry => entry.s0_raw || 0),
                        s1_raw: recent20sData.map(entry => entry.s1_raw || 0)
                    };

                    try {
                        const response = await fetch('/estimate_bp', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(requestData)
                        });
                        const result = await response.json();

                        // 4. 成功→更新数值显示+图表
                        if (result.success) {
                            const currentElapsed = (new Date().getTime() - startTimestamp) / 1000; // 当前相对时间
                            const sbp = parseFloat(result.sbp.toFixed(0));
                            const dbp = parseFloat(result.dbp.toFixed(0));

                            // 更新文字显示
                            estSbpElem.textContent = sbp;
                            estDbpElem.textContent = dbp;
                            estStatusElem.textContent = `BP Updated (Data used: ${windowStartTime.toFixed(1)}-${latestTime.toFixed(1)}s)`;
                            estStatusElem.style.color = 'green';

                            // 更新血压图表
                            updateBpChart(currentElapsed, sbp, dbp);
                        } 
                        // 失败→仅更新状态文字
                        else {
                            estStatusElem.textContent = `BP Estimation Error: ${result.message}`;
                            estStatusElem.style.color = 'red';
                        }
                    } catch (err) {
                        estStatusElem.textContent = 'BP Estimation: Server connection failed';
                        estStatusElem.style.color = 'red';
                        console.error('Auto BP error:', err);
                    }
                } 
                function updateBpChart(elapsedTime, sbp, dbp) {
                    // 1. 添加新数据（x轴：相对时间，y轴：SBP/DBP）
                    bpChartData.labels.push(elapsedTime.toFixed(1));
                    bpChartData.datasets[0].data.push(sbp);  // SBP数据
                    bpChartData.datasets[1].data.push(dbp);  // DBP数据

                    // 2. 限制图表点数（超过MAX_BP_POINTS则删除最旧数据）
                    if (bpChartData.labels.length > MAX_BP_POINTS) {
                        bpChartData.labels.shift();  // 删除最旧时间
                        bpChartData.datasets[0].data.shift();  // 删除最旧SBP
                        bpChartData.datasets[1].data.shift();  // 删除最旧DBP
                    }

                    // 3. 刷新图表
                    bpChart.update();
                }                 

                // 倒计时显示
                function startCountdown() {
                    let remaining = AUTO_STOP_DURATION / 1000;
                    updateTimerDisplay(remaining);

                    countdownInterval = setInterval(function() {
                        remaining--;
                        updateTimerDisplay(remaining);
                        if (remaining <= 0) {
                            clearInterval(countdownInterval);
                        }
                    }, 1000);
                }

                function updateTimerDisplay(seconds) {
                    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
                    const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
                    timerElem.textContent = `Auto-stop Countdown: ${mins}:${secs}`;
                }

                // 数据保存（CSV）
                saveBtn.addEventListener('click', function() {
                if (rawDataHistory.length === 0) {
                    alert('No data available to save!');
                    return;
                }

                // 1. 弹出输入框让用户自定义文件名（默认带时间戳，避免重名）
                const defaultFileName = 'ppg_data_' + new Date().toISOString().replace(/:/g, '-');
                const userFileName = prompt('Please enter the filename to save (without .csv suffix):', defaultFileName);
                
                // 2. 如果用户取消输入，直接返回
                if (userFileName === null) {
                    return;
                }
                
                // 3. 处理文件名（去除非法字符，添加.csv后缀）
                const validFileName = userFileName
                    .replace(/[\\/:*?"<>|]/g, '')  // 移除操作系统不允许的特殊字符
                    .trim() || defaultFileName;  // 若输入为空，用默认名
                const fullFileName = validFileName + '.csv';  // 强制添加.csv后缀

                // 4. 生成CSV内容并触发下载（浏览器会自动弹出路径选择弹窗）
                let csvContent = "timestamp,seconds,s0_raw,s1_raw,s0_bpm,s1_bpm,SBP,DBP\n";
                rawDataHistory.forEach(function(entry) {
                    csvContent += `${entry.timestamp},${entry.seconds.toFixed(3)},${entry.s0_raw || ''},${entry.s1_raw || ''},${entry.s0_bpm},${entry.s1_bpm},${sbpValue},${dbpValue}\n`;
                });
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = fullFileName;  // 使用用户输入的文件名
                document.body.appendChild(link);
                link.click();  // 触发下载（浏览器会弹出保存路径选择窗口）
                document.body.removeChild(link);
                URL.revokeObjectURL(url);  // 释放URL对象，避免内存泄漏
            });

                // -------------------------- RAW图表（最多500个点） --------------------------
                const rawCtx = document.getElementById('raw-chart').getContext('2d');
                const rawChartData = {
                    labels: [],  // 横坐标：相对秒数（精确到3位）
                    datasets: [
                        { label: 'Sensor 0 Raw', data: [], borderColor: 'rgb(54, 162, 235)', borderWidth: 1, tension: 0.1, fill: false, pointRadius: 0, pointHoverRadius: 0},
                        { label: 'Sensor 1 Raw', data: [], borderColor: 'rgb(255, 99, 132)', borderWidth: 1, tension: 0.1, fill: false, pointRadius: 0, pointHoverRadius: 0}
                    ]
                };
                const rawChart = new Chart(rawCtx, {
                    type: 'line',
                    data: rawChartData,
                    options: {
                        responsive: true,
                        animation: false,
                        scales: {
                            x: { 
                                title: { display: true, text: 'Time (s)' },
                                ticks: { stepSize: 0.5 }
                            },
                            y: { title: { display: true, text: 'Raw AD Value (0-4095)' }, min: 0, max: 4095 }
                        }
                    }
                });

                // BPM图表
                const bpmCtx = document.getElementById('bpm-chart').getContext('2d');
                const bpmChartData = {
                    labels: [],
                    datasets: [
                        { label: 'Sensor 0 BPM', data: [], borderColor: 'rgb(54, 162, 235)', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 4 },
                        { label: 'Sensor 1 BPM', data: [], borderColor: 'rgb(255, 99, 132)', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 4 }
                    ]
                };
                const bpmChart = new Chart(bpmCtx, {
                    type: 'line',
                    data: bpmChartData,
                    options: {
                        responsive: true, animation: false,
                        scales: {
                            x: { title: { display: true, text: 'Time (s)' } },
                            y: { title: { display: true, text: 'Heart Rate (BPM)' }, min: 40, max: 120 }
                        }
                    }
                }); 
                // BP chart 
                const bpCtx = document.getElementById('bp-chart').getContext('2d');
                // 1. 定义图表数据（与BPM/RAW一样，在初始化时直接声明）
                const bpChartData = {
                    labels: [],  // x轴：相对监控开始时间（秒）
                    datasets: [
                        {
                            label: 'SBP (mmHg)',
                            data: [],
                            borderColor: 'rgb(255, 99, 132)',  // 红色SBP
                            borderWidth: 2,
                            tension: 0.3,
                            fill: false,
                            pointRadius: 4
                        },
                        {
                            label: 'DBP (mmHg)',
                            data: [],
                            borderColor: 'rgb(54, 162, 235)',  // 蓝色DBP
                            borderWidth: 2,
                            tension: 0.3,
                            fill: false,
                            pointRadius: 4
                        }
                    ]
                };
                // 2. 创建图表实例（直接引用上面定义的bpChartData）
                const bpChart = new Chart(bpCtx, {
                    type: 'line',
                    data: bpChartData,
                    options: {
                        responsive: true,
                        animation: false,
                        scales: {
                            x: {
                                title: { display: true, text: 'Time Since Monitoring Start (s)' },
                                ticks: { stepSize: 5 }
                            },
                            y: {
                                title: { display: true, text: 'Blood Pressure (mmHg)' },
                                min: 40,
                                max: 180,
                                ticks: { stepSize: 20 }
                            }
                        },
                        plugins: { legend: { position: 'top' } }
                    }
                });            

                // 重置图表
                function resetCharts() {
                    // BPM图表重置
                    bpmChartData.labels = [];
                    bpmChartData.datasets.forEach(ds => ds.data = []);
                    bpmChart.update();

                    // RAW图表重置
                    rawChartData.labels = [];
                    rawChartData.datasets.forEach(ds => ds.data = []);
                    rawChart.update();
                                
                    // BP图表重置
                    bpChartData.labels = [];
                    bpChartData.datasets.forEach(ds => ds.data = []);
                    bpChart.update();
                }

                // -------------------------- 接收实时数据（修复时间戳问题） --------------------------
                socket.on('update', function(data) {
                    if (!monitoringActive || !startTimestamp) return;

                    const currentBatchTimestamp = new Date().getTime();  // 接收当前批量数据的系统时间（毫秒）
                    const s0 = parseFloat(data.bpm?.s0) || 0;
                    const s1 = parseFloat(data.bpm?.s1) || 0;
                    const s0RawList = data.raw?.s0_raw || [];
                    const s1RawList = data.raw?.s1_raw || [];
                    const rawCount = Math.max(s0RawList.length, s1RawList.length);

                    // 1. 更新数值显示
                    document.getElementById('s0-bpm').textContent = s0.toFixed(0);
                    document.getElementById('s1-bpm').textContent = s1.toFixed(0);
                    if (s0RawList.length > 0) document.getElementById('s0-raw').textContent = s0RawList[s0RawList.length - 1].toFixed(0);
                    if (s1RawList.length > 0) document.getElementById('s1-raw').textContent = s1RawList[s1RawList.length - 1].toFixed(0);

                    // 2. RAW数据处理（核心修复：强制时间戳≥startTimestamp）
                    if (rawCount > 0) {
                        const batchDuration = Math.max(rawCount * SAFE_INTERVAL, 1);  // 最小1ms避免异常
                        
                        let batchStartTime = currentBatchTimestamp - batchDuration;
                        if (globalLastTimestamp !== null) {
                            // 强制起始时间 ≥ 上一批最后一个点时间（确保跨批次递增）
                            batchStartTime = Math.max(batchStartTime, startTimestamp, globalLastTimestamp);
                        } else {
                            batchStartTime = Math.max(batchStartTime, startTimestamp);
                        }


                        // 修复2：重新计算批量总时长（如果起始时间被修正，避免时间戳重叠）
                        const actualBatchDuration = Math.max(currentBatchTimestamp - batchStartTime, 1);

                        for (let i = 0; i < rawCount; i++) {
                            // 每个点的精确时间戳（毫秒）：基于修正后的起始时间和实际时长
                            const pointTimestamp = batchStartTime + (i * actualBatchDuration) / (rawCount - 1 || 1);
                            
                            // 再次确保时间戳不早于基准时间（双重保险）
                            const safePointTimestamp = Math.max(pointTimestamp, globalLastTimestamp || startTimestamp);
                                  
                            globalLastTimestamp = safePointTimestamp;      
                            
                            // 相对时间（秒）=（安全时间戳 - 基准时间）/ 1000（确保≥0）
                            const relativeSec = (safePointTimestamp - startTimestamp) / 1000;

                            // 添加新数据到图表
                            rawChartData.labels.push(relativeSec.toFixed(1));
                            rawChartData.datasets[0].data.push(s0RawList[i] || null);
                            rawChartData.datasets[1].data.push(s1RawList[i] || null);

                            // 记录历史数据
                            rawDataHistory.push({
                                timestamp: new Date(safePointTimestamp).toISOString(),
                                seconds: relativeSec,  // 此时seconds一定≥0
                                s0_raw: s0RawList[i] || null,
                                s1_raw: s1RawList[i] || null,
                                s0_bpm: s0,
                                s1_bpm: s1
                            });
                        }
                    }

                    // 超过500个点时，移除最旧的数据
                    if (rawChartData.labels.length > MAX_RAW_POINTS) {
                        const excess = rawChartData.labels.length - MAX_RAW_POINTS;
                        rawChartData.labels.splice(0, excess);
                        rawChartData.datasets[0].data.splice(0, excess);
                        rawChartData.datasets[1].data.splice(0, excess);
                    }
                    rawChart.update();

                    // 3. BPM图表更新
                    const currentRelativeSec = (new Date().getTime() - startTimestamp) / 1000;
                    const isNewBpmPush = data.raw?.s0_raw?.length === 0 && data.raw?.s1_raw?.length === 0;
                                  
                    if (
                    bpmChartData.labels.length === 0 
                    || 
                    (
                        isNewBpmPush && 
                        currentRelativeSec - parseFloat(bpmChartData.labels[bpmChartData.labels.length - 1]) >= 1
                    )
                    ) {
                        bpmChartData.labels.push(currentRelativeSec.toFixed(1));
                        bpmChartData.datasets[0].data.push(s0);
                        bpmChartData.datasets[1].data.push(s1);
                        if (bpmChartData.labels.length > MAX_BPM_POINTS) {
                            bpmChartData.labels.shift();
                            bpmChartData.datasets.forEach(ds => ds.data.shift());
                        }
                        bpmChart.update();
                    }
                });

                // Socket连接状态
                socket.on('connect', () => {
                    console.log('✅ Socket connected');
                    const statusText = statusElem.textContent;
                    if (statusText.includes('(Connected to Server)')) {
                        statusText = statusText.replace(' (Connected to Server)', '');
                    }
                    statusElem.textContent = statusText + ' (Connected to Server)';
                });
                socket.on('connect_error', (err) => console.log('❌ Connection error:', err));
            });
        </script>
    </body>
    </html>
    """)


# 处理控制命令（开始/停止推送）
@socketio.on('control')
def handle_control(data):
    global is_pushing, start_time, raw_data_history, latest_bpm
    action = data.get('action')
    print(f"Control command received: {action}")
    
    if action == 'start':
        is_pushing = True
        start_time = datetime.now()
        raw_data_history = []
        latest_bpm = {'s0': 0.0, 's1': 0.0}
        emit('control_response', {'status': 'started'})
    elif action == 'stop':
        is_pushing = False
        emit('control_response', {'status': 'stopped'})


# 处理批量RAW数据（实时推送）
@app.route('/raw', methods=['POST'])
def receive_raw_data():
    global is_pushing
    if not is_pushing:
        return "OK"
    
    s0_raw_list = [int(val) for val in request.form.getlist('s0_raw') if val.strip()]
    s1_raw_list = [int(val) for val in request.form.getlist('s1_raw') if val.strip()]

    
    def clean_raw(val):
        return max(0, min(4095, val))
    cleaned_s0 = [clean_raw(val) for val in s0_raw_list]
    cleaned_s1 = [clean_raw(val) for val in s1_raw_list]
    
    socketio.emit('update', {
        'raw': {'s0_raw': cleaned_s0, 's1_raw': cleaned_s1},
        'bpm': latest_bpm
    })
    return "OK"


# 处理BPM数据
@app.route('/bpm', methods=['POST'])
def receive_bpm_data():
    global latest_bpm, is_pushing
    if not is_pushing:
        return "OK"
    
    form_data = request.form.to_dict()
    def clean_bpm(val):
        try:
            return max(40, min(120, float(val)))
        except:
            return None
    
    if 's0' in form_data:
        cleaned = clean_bpm(form_data['s0'])
        if cleaned is not None:
            latest_bpm['s0'] = cleaned
    if 's1' in form_data:
        cleaned = clean_bpm(form_data['s1'])
        if cleaned is not None:
            latest_bpm['s1'] = cleaned
    
    socketio.emit('update', {
        'raw': {'s0_raw': [], 's1_raw': []},
        'bpm': latest_bpm
    })
    return "OK"

@app.route('/bp', methods=['POST'])
def receive_bp_data():
    global latest_bp
    form_data = request.form.to_dict()
    # 清洗数据（确保为数字，默认0）
    def clean_bp(val):
        try:
            return float(val) if val.strip() else 0.0
        except:
            return 0.0
    # 更新全局变量
    if 'sbp' in form_data:
        latest_bp['sbp'] = clean_bp(form_data['sbp'])
    if 'dbp' in form_data:
        latest_bp['dbp'] = clean_bp(form_data['dbp'])
    return "OK"

@app.route('/estimate_bp', methods=['POST'])
def estimate_bp():
    try:
        # 步骤1：接收前端发送的JSON数据
        request_data = json.loads(request.data)
        seconds = request_data.get('seconds', [])
        s0_raw = request_data.get('s0_raw', [])
        s1_raw = request_data.get('s1_raw', [])

        # 步骤2：数据有效性检查（长度一致）
        if len(seconds) != len(s0_raw) or len(seconds) != len(s1_raw):
            return json.dumps({
                'success': False,
                'message': 'Data length mismatch (seconds/s0_raw/s1_raw must have same length)'
            })

        # 步骤3：转换数据格式（根据PPG_feature_extraction函数需求，通常转为numpy数组）
        seconds_np = np.array(seconds, dtype=np.float64)
        s0_raw_np = np.array(s0_raw, dtype=np.float64)
        s1_raw_np = np.array(s1_raw, dtype=np.float64)

        # 步骤4：调用PPG_feature_extraction函数（假设函数返回SBP和DBP）
        # 注意：需与函数实际返回值匹配！若函数返回多个值，需调整解构方式
        # 示例：假设函数返回 (PTT, HR, PWV, SBP, DBP)，取后两个值
        PTT, HR, PWV = PPG_feature_extraction(seconds_np, s0_raw_np, s1_raw_np)
        estimated_sbp, estimated_dbp = BP_calculation(PTT, HR, PWV)
        
        # 步骤5：返回成功结果（限制血压范围，避免异常值）
        estimated_sbp = max(60, min(180, estimated_sbp))  # 合理SBP范围
        estimated_dbp = max(40, min(120, estimated_dbp))  # 合理DBP范围
        return json.dumps({
            'success': True,
            'sbp': estimated_sbp,
            'dbp': estimated_dbp
        })

    except Exception as e:
        # 捕获异常，返回错误信息
        return json.dumps({
            'success': False,
            'message': f'Calculation failed: {str(e)}'
        })

# 服务器端数据导出（备用）
@app.route('/export_data')
def export_data():
    global raw_data_history
    if not raw_data_history:
        return "No data available", 400
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['timestamp', 'seconds', 's0_raw', 's1_raw', 's0_bpm', 's1_bpm', 'SBP', 'DBP'])
    current_sbp = latest_bp['sbp']
    current_dbp = latest_bp['dbp']
    for entry in raw_data_history:
        writer.writerow([
            entry['timestamp'], entry['seconds'],
            entry['s0_raw'], entry['s1_raw'],
            entry['s0_bpm'], entry['s1_bpm'],
            current_sbp, current_dbp
        ])
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=ppg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response.headers["Content-type"] = "text/csv"
    return response


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)