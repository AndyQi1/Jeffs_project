from flask import Flask, request, render_template, make_response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import csv
from io import StringIO
from datetime import datetime
import json
import numpy as np  # 需要导入numpy处理数据
# 确保 function_lib 中有这两个函数
from function_lib import PPG_feature_extraction, BP_calculation 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# ---------------------- 全局变量 ----------------------
# 存储最新的BPM值和BP值
latest_bpm = {'s0': 0.0, 's1': 0.0}
latest_bp = {'sbp': 0.0, 'dbp': 0.0}

raw_data_history = []  
is_pushing = False     
start_time = None

@app.route('/')
def index():
    return render_template('index.html')

# ---------------------- 控制逻辑 ----------------------
@socketio.on('control')
def handle_control(data):
    global is_pushing, start_time, raw_data_history, latest_bpm, latest_bp
    action = data.get('action')
    print(f"Control command received: {action}")
    
    if action == 'start':
        is_pushing = True
        start_time = datetime.now()
        raw_data_history = []
        # 重置数值
        latest_bpm = {'s0': 0.0, 's1': 0.0}
        latest_bp = {'sbp': 0.0, 'dbp': 0.0}
        emit('control_response', {'status': 'started'})
    elif action == 'stop':
        is_pushing = False
        emit('control_response', {'status': 'stopped'})

# ---------------------- 核心修改 1: /raw 接口 ----------------------
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
    
    # 【修改点】: 在推送波形的同时，推送全局变量中存储的最新的 BPM 和 BP
    socketio.emit('update', {
        'raw': {'s0_raw': cleaned_s0, 's1_raw': cleaned_s1},
        'bpm': latest_bpm, # 包含 s0, s1 的心率
        'bp': latest_bp    # 包含 sbp, dbp 的血压
    })
    return "OK"

# ---------------------- 核心修改 2: 新增统一计算接口 ----------------------
@app.route('/estimate_all', methods=['POST'])
def estimate_all():
    global latest_bpm, latest_bp
    try:
        # 1. 接收数据
        request_data = json.loads(request.data)
        seconds = request_data.get('seconds', [])
        s0_raw = request_data.get('s0_raw', [])
        s1_raw = request_data.get('s1_raw', [])

        # 2. 校验长度
        if len(seconds) != len(s0_raw) or len(seconds) != len(s1_raw):
            return json.dumps({'success': False, 'message': 'Data length mismatch'})
        
        if len(seconds) < 100: # 数据太少不计算
            return json.dumps({'success': False, 'message': 'Not enough data'})

        # 3. 转为 numpy 数组
        seconds_np = np.array(seconds, dtype=np.float64)
        s0_raw_np = np.array(s0_raw, dtype=np.float64)
        s1_raw_np = np.array(s1_raw, dtype=np.float64)

        # 4. 调用算法 (同时算出 BPM 和 特征值)
        # 假设 PPG_feature_extraction 返回: (PTT, HR, PWV)
        try:
            PTT, HR, PWV, s0_proc, s1_proc = PPG_feature_extraction(seconds_np, s0_raw_np, s1_raw_np)
        except Exception as e:
            print(f"Algorithm Error: {e}")
            return json.dumps({'success': False, 'message': 'Feature extraction failed'})

        # 5. 计算血压
        # 假设 BP_calculation 使用上面算出的特征
        try:
            estimated_sbp, estimated_dbp = BP_calculation(PTT, HR, PWV)
        except Exception as e:
             print(f"BP Algo Error: {e}")
             return json.dumps({'success': False, 'message': 'BP calculation failed'})

        # 6. 数据清洗 (NaN检查 + 范围限制)
        def safe_val(val, min_v, max_v):
            if np.isnan(val) or np.isinf(val): return 0
            return int(max(min_v, min(max_v, float(val))))

        clean_hr = safe_val(HR, 40, 180)
        clean_sbp = safe_val(estimated_sbp, 60, 180)
        clean_dbp = safe_val(estimated_dbp, 40, 120)

        # 7. 更新全局变量 (关键步骤：这里更新后，/raw 接口推送时就会带上新值)
        latest_bpm['s0'] = clean_hr
        latest_bpm['s1'] = clean_hr # 假设双路心率一致
        
        latest_bp['sbp'] = clean_sbp
        latest_bp['dbp'] = clean_dbp

        # 8. 返回结果给前端 (用于更新文字显示)
        return json.dumps({
            'success': True,
            'bpm': int(HR), # 这里填你清洗后的变量
            'sbp': int(estimated_sbp),
            'dbp': int(estimated_dbp),
            
            # 【关键修改】将 numpy 数组转换为 list 返回给前端
            # 如果 s0_proc 是 numpy array，必须用 .tolist()
            's0_proc': s0_proc.tolist(),
            's1_proc': s1_proc.tolist()
        })

    except Exception as e:
        print(f"General Error: {e}")
        return json.dumps({
            'success': False,
            'message': f'Calculation failed: {str(e)}'
        })

# ---------------------- 辅助接口 ----------------------
# 接收真实血压输入（用于校准或记录）
@app.route('/bp', methods=['POST'])
def receive_bp_data():
    # 注意：这里主要用于接收用户在网页填写的“真实血压”
    # 如果你想把它和估算血压区分开，建议不要覆盖 latest_bp，而是用另一个变量
    # 但如果你想覆盖，逻辑如下：
    form_data = request.form.to_dict()
    val = 0.0
    if 'sbp' in form_data:
        try: val = float(form_data['sbp'])
        except: val = 0.0
        # latest_bp['sbp'] = val # 取决于你的需求，是否要覆盖算法值
    if 'dbp' in form_data:
        try: val = float(form_data['dbp'])
        except: val = 0.0
        # latest_bp['dbp'] = val
    return "OK"

@app.route('/export_data')
def export_data():
    global raw_data_history
    if not raw_data_history:
        return "No data available", 400
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['timestamp', 'seconds', 's0_raw', 's1_raw', 's0_bpm', 's1_bpm', 'SBP', 'DBP'])
    
    # 这里的导出使用的是服务器端历史数据
    # 注意：服务器端历史数据可能不如前端完整，建议主要使用前端导出
    for entry in raw_data_history:
        writer.writerow([
            entry.get('timestamp', ''), 
            entry.get('seconds', 0),
            entry.get('s0_raw', 0), 
            entry.get('s1_raw', 0),
            entry.get('s0_bpm', 0), 
            entry.get('s1_bpm', 0),
            entry.get('sbp', 0), 
            entry.get('dbp', 0)
        ])
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=ppg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response.headers["Content-type"] = "text/csv"
    return response

if __name__ == '__main__':
    # 允许所有IP访问
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)