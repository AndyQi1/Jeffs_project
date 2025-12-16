document.addEventListener('DOMContentLoaded', function () {
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

    let isEstimating = false;    // 【修改】改名：是否处于分析模式
    let estimateInterval = null; // 【修改】改名：分析定时器
    const UPDATE_INTERVAL = 1000; // 【修改】改为2秒刷新一次，频率更高


    let globalLastTimestamp = null;  // 全局最后一个点的时间戳（确保递增）

    let sbpValue = 0;
    let dbpValue = 0;
    sbpInput.addEventListener('input', function () {
        sbpValue = parseFloat(this.value) || 0;
        // 发送到服务器
        fetch('/bp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `sbp=${sbpValue}`
        });
    });
    dbpInput.addEventListener('input', function () {
        dbpValue = parseFloat(this.value) || 0;
        // 发送到服务器
        fetch('/bp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `dbp=${dbpValue}`
        });
    });


    const THEORETICAL_SAMPLE_RATE = 500;  // 理论采样率：500Hz
    const BASE_INTERVAL = 1000 / THEORETICAL_SAMPLE_RATE;  // 理论间隔：2ms
    const ACTUAL_INTERVAL_FACTOR = 1.2;  // 实际间隔系数（留1.2倍弹性，适配采样率降低）
    const SAFE_INTERVAL = BASE_INTERVAL * ACTUAL_INTERVAL_FACTOR;  // 安全间隔：4.8ms（兼容实际波动）

    // 开始监控逻辑（核心：记录基准时间）
    startBtn.addEventListener('click', function () {
        monitoringActive = true;
        estimateBtn.disabled = false;
        estimateBtn.textContent = 'Start Analysis';  // 初始文字              
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
        autoStopTimer = setTimeout(function () {
            if (monitoringActive) {
                statusElem.textContent = 'Status: Auto-stop (2 min reached)';
                stopMonitoring();
            }
        }, AUTO_STOP_DURATION);

        // 启动倒计时
        startCountdown();
    });

    // 停止监控逻辑
    stopBtn.addEventListener('click', function () {
        statusElem.textContent = 'Status: Manual Stop';
        stopMonitoring();

    });

    function stopMonitoring() {
        monitoringActive = false;
        socket.emit('control', { action: 'stop' });
        // 新增：停止BP测量（若正在运行）
        // 【修改】停止统一分析逻辑
        if (isEstimating) {
            isEstimating = false;
            clearInterval(estimateInterval);
            estimateBtn.textContent = 'Start Analysis';
            estStatusElem.textContent = 'Analysis Stopped (Monitoring Ended)';
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
        estStatusElem.textContent = 'Click "Start Analysis" to calculate';
    }

    estimateBtn.addEventListener('click', function () {
        if (!monitoringActive) {
            estStatusElem.textContent = 'Error: Start monitoring first!';
            estStatusElem.style.color = 'red';
            return;
        }

        // 状态1：未开始 -> 启动
        if (!isEstimating) {
            isEstimating = true;
            estimateBtn.textContent = 'Stop Analysis'; // 【修改】文案
            estimateBtn.classList.add('running');
            estStatusElem.textContent = 'Analysis Running (BPM + BP)';
            estStatusElem.style.color = 'green';

            // 立即计算一次，再启动定时
            autoCalculateMetrics();
            estimateInterval = setInterval(autoCalculateMetrics, UPDATE_INTERVAL);
        }
        // 状态2：已开始 -> 停止
        else {
            isEstimating = false;
            estimateBtn.textContent = 'Start Analysis'; // 【修改】文案
            estimateBtn.classList.remove('running');
            estStatusElem.textContent = 'Analysis Stopped';
            estStatusElem.style.color = 'orange';

            // 清除定时任务
            clearInterval(estimateInterval);
        }
    });
    // 【核心修改】统一计算函数：调用 /estimate_all
    async function autoCalculateMetrics() {
        if (!monitoringActive || !isEstimating) return;

        // 1. 筛选最近 10秒 数据（BPM计算通常10秒响应更快）
        const latestTime = rawDataHistory[rawDataHistory.length - 1]?.seconds || 0;
        const windowStartTime = Math.max(0, latestTime - 10);
        const recentData = rawDataHistory.filter(entry =>
            entry.seconds >= windowStartTime && entry.seconds <= latestTime
        );

        // 2. 数据不足处理
        if (recentData.length < 200) { // 约0.4秒的数据量
            estStatusElem.textContent = `Buffering data... (${recentData.length})`;
            estStatusElem.style.color = 'orange';
            return;
        }

        // 3. 构建请求数据
        const requestData = {
            seconds: recentData.map(entry => entry.seconds),
            s0_raw: recentData.map(entry => entry.s0_raw || 0),
            s1_raw: recentData.map(entry => entry.s1_raw || 0)
        };

        try {
            // 【关键】调用新的统一接口
            const response = await fetch('/estimate_all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });
            const result = await response.json();

            // 4. 处理返回结果
            if (result.success) {
                const currentElapsed = (new Date().getTime() - startTimestamp) / 1000;

                // 获取返回的 BPM 和 BP
                const bpm = parseFloat(result.bpm);
                const sbp = parseFloat(result.sbp);
                const dbp = parseFloat(result.dbp);

                // 使用 .toFixed(0) 取整，或者 .toFixed(1) 保留一位小数
                estSbpElem.textContent = sbp.toFixed(0);
                estDbpElem.textContent = dbp.toFixed(0);
                estStatusElem.textContent = 'Analyzing'; 
                estStatusElem.style.color = 'green';

                // 更新血压图表
                updateBpChart(currentElapsed, sbp, dbp);
                // 我们直接全量替换图表数据，显示当前分析窗口的波形状态
                if (result.s0_proc && result.s1_proc) {
                    // 使用发送请求时的时间轴作为 X 轴
                    // 为了显示效果，我们可以只截取小数点后1位
                    const timeLabels = requestData.seconds.map(t => t.toFixed(1));
                    
                    procChartData.labels = timeLabels;
                    procChartData.datasets[0].data = result.s0_proc;
                    procChartData.datasets[1].data = result.s1_proc;
                    
                    procChart.update();
                }
                
                // 注意：BPM图表不需要在这里更新，后端已更新全局变量，Socket会自动推送
            } else {
                estStatusElem.textContent = `Error: ${result.message}`;
                estStatusElem.style.color = 'red';
            }
        } catch (err) {
            console.error('Auto metrics error:', err);
            estStatusElem.textContent = 'Analysis Error: Server connection failed';
            estStatusElem.style.color = 'red';
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

        countdownInterval = setInterval(function () {
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
    saveBtn.addEventListener('click', function () {
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
        rawDataHistory.forEach(function (entry) {
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
            { label: 'Sensor 0 Raw', data: [], borderColor: 'rgb(54, 162, 235)', borderWidth: 1, tension: 0.1, fill: false, pointRadius: 0, pointHoverRadius: 0 },
            { label: 'Sensor 1 Raw', data: [], borderColor: 'rgb(255, 99, 132)', borderWidth: 1, tension: 0.1, fill: false, pointRadius: 0, pointHoverRadius: 0 }
        ]
    };
    const rawChart = new Chart(rawCtx, {
        type: 'line',
        data: rawChartData,
        options: {
            maintainAspectRatio: false,
            responsive: true,
            animation: false,
            scales: {
                x: {
                    title: { display: true, text: 'Time (s)' },
                    ticks: { 
                        // 【关键修改】删掉了 stepSize: 0.5
                        maxTicksLimit: 6,    // 强制最多显示 6 个刻度
                        maxRotation: 0,      // 禁止旋转
                        autoSkip: true,      // 允许自动跳过拥挤的标签
                        font: { size: 11 }    // 字体改小
                    }
                },
                y: { title: { display: true, text: 'Raw AD Value (0-4095)' }, min: 0, max: 4095 }
            }
        }
    });

    // 【新增】Preprocessed Data 图表初始化
    const procCtx = document.getElementById('processed-chart').getContext('2d');
    const procChartData = {
        labels: [], 
        datasets: [
            { label: 'S0 Proc', data: [], borderColor: 'rgb(75, 192, 192)', borderWidth: 1, tension: 0.1, pointRadius: 0 },
            { label: 'S1 Proc', data: [], borderColor: 'rgb(153, 102, 255)', borderWidth: 1, tension: 0.1, pointRadius: 0 }
        ]
    };
    const procChart = new Chart(procCtx, {
        type: 'line',
        data: procChartData,
        options: {
            maintainAspectRatio: false,
            responsive: true,
            animation: false, // 关闭动画以获得高性能刷新
            scales: {
                x: { 
                    display: true, 
                    title: { display: true, text: 'Time (s)' },
                    ticks: { 
                        // 【关键修改】删掉了 stepSize: 0.5
                        maxTicksLimit: 6,    // 强制最多显示 6 个刻度
                        maxRotation: 0,      // 禁止旋转
                        autoSkip: true,      // 允许自动跳过拥挤的标签
                        font: { size: 11 }    // 字体改小
                    } // 限制x轴标签数量，防止重叠
                },
                y: { display: true, title: { display: true, text: 'Amplitude' } }
            }
        }
    });

    // BPM图表
    const bpmCtx = document.getElementById('bpm-chart').getContext('2d');
    
    // 创建红色渐变背景
    const bpmGradient = bpmCtx.createLinearGradient(0, 0, 0, 220);
    bpmGradient.addColorStop(0, 'rgba(255, 99, 132, 0.5)'); 
    bpmGradient.addColorStop(1, 'rgba(255, 99, 132, 0)');   

    const bpmChartData = {
        labels: [],
        datasets: [
            { 
                label: 'Heart Rate', 
                data: [], 
                borderColor: 'rgb(255, 99, 132)', 
                backgroundColor: bpmGradient,     
                borderWidth: 2, 
                tension: 0.4,       // 平滑曲线
                fill: true,         // 填充渐变
                pointRadius: 0,     
                pointHoverRadius: 4
            }
        ]
    };
    
    const bpmChart = new Chart(bpmCtx, {
        type: 'line',
        data: bpmChartData,
        options: {
            maintainAspectRatio: false,
            responsive: true, 
            animation: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: true } 
            },
            scales: {
                x: { 
                    display: true, 
                    title: { 
                        display: true, 
                        text: 'Time (s)', 
                        // 【修改】字体改回 12px (或者你也可以删掉 font 这一行用默认)
                        font: { size: 12 }, 
                        padding: { top: 0 }
                    }, 
                    ticks: { 
                        maxTicksLimit: 6, 
                        // 【修改】刻度字体也改回 11px 或 12px
                        font: { size: 11 }, 
                        maxRotation: 0
                    },
                    grid: { display: false }
                },
                y: { 
                    display: true,
                    min: 40, max: 120,
                    title: { 
                        display: true, 
                        text: 'Heart Rate (BPM)', 
                        // 【修改】字体改回 12px
                        font: { size: 12 } 
                    },
                    grid: { color: '#f5f5f5' },
                    ticks: { 
                        // 【修改】刻度字体也改回 11px 或 12px
                        font: { size: 11 } 
                    }
                }
            },
            layout: {
                padding: { bottom: 0, left: 0, right: 10 }
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
            maintainAspectRatio: false,
            responsive: true,
            animation: false,
            scales: {
                x: {
                    title: { display: true, text: 'Time (s)' },
                    ticks: { 
                        // 【关键修改】删掉了 stepSize: 0.5
                        maxTicksLimit: 6,    // 强制最多显示 6 个刻度
                        maxRotation: 0,      // 禁止旋转
                        autoSkip: true,      // 允许自动跳过拥挤的标签
                        font: { size: 9 }    // 字体改小
                    }
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

        // 【新增】重置处理后的图表
        procChartData.labels = [];
        procChartData.datasets.forEach(ds => ds.data = []);
        procChart.update();

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
    socket.on('update', function (data) {
        if (!monitoringActive || !startTimestamp) return;

        const currentBatchTimestamp = new Date().getTime();  // 接收当前批量数据的系统时间（毫秒）
        const s0 = parseFloat(data.bpm?.s0) || 0;
        const s1 = parseFloat(data.bpm?.s1) || 0;
        // 【新增】从 data.bp 获取血压数据 (实时流带过来的最新计算值)
        const sbp = parseFloat(data.bp?.sbp) || 0;
        const dbp = parseFloat(data.bp?.dbp) || 0;
        const s0RawList = data.raw?.s0_raw || [];
        const s1RawList = data.raw?.s1_raw || [];
        const rawCount = Math.max(s0RawList.length, s1RawList.length);

        // 1. 更新数值显示
        document.getElementById('s0-bpm').textContent = s0.toFixed(0);
        document.getElementById('s1-bpm').textContent = s1.toFixed(0);
        // 【新增】如果正在进行估算，确保显示的是最新的同步值
        if (isEstimating) {
            // 如果大于0，显示取整后的值；否则显示 '--'
            document.getElementById('est-sbp').textContent = sbp > 0 ? sbp.toFixed(0) : '--';
            document.getElementById('est-dbp').textContent = dbp > 0 ? dbp.toFixed(0) : '--';
        }
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

        // ... (socket.on 内部的前面代码保持不变) ...

        // 3. BPM图表更新
        const currentRelativeSec = (new Date().getTime() - startTimestamp) / 1000;
        
        if (isEstimating) {
            if (
                bpmChartData.labels.length === 0 || 
                (currentRelativeSec - parseFloat(bpmChartData.labels[bpmChartData.labels.length - 1]) >= 1)
            ) {
                bpmChartData.labels.push(currentRelativeSec.toFixed(1));
                
                // 【修改】只推送 s0 的值（因为 s0 和 s1 的 BPM 现在是一样的）
                bpmChartData.datasets[0].data.push(s0); 
                
                // 【删除】删掉了 bpmChartData.datasets[1].data.push(s1);

                if (bpmChartData.labels.length > MAX_BPM_POINTS) {
                    bpmChartData.labels.shift();
                    bpmChartData.datasets[0].data.shift(); // 【修改】只移除第一组数据
                }
                bpmChart.update();
            }
        }
    }); // 结束 socket.on

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