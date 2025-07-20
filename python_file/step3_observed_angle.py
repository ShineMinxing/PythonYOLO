"""
step3_observed_angle.py  •  v3.1  (YOLO label generator with area filter)
----------------------------------------------------------------
为裁剪后视频中的无人机轨迹绘制可视化箭头与包围框，并同时：
1. 生成符合 YOLO 训练格式的 .txt 标签文件（附加 front/side 角）。
2. 将对应的图像帧 (.jpg) 导出到 images 目录（支持 track_area 过滤）。
3. 输出带包围框可视化的 Observed_*.mp4（不受过滤影响）。

💡 YOLO 标签格式（每行一目标）：
    class x1 y1 ... x4 y4 kp_x kp_y kp_v
  * class 固定为 0（drone）。
"""
import os, cv2, yaml, subprocess, pandas as pd, math, numpy as np
from tqdm import tqdm

# ---------------------------- 默认配置 ---------------------------- #
DEFAULT_CFG = {
    'raw_csv_dir':     'local_file/raw_file',
    'input_video_dir': 'local_file/step2_file',
    'input_csv_dir':   'local_file/step2_file',
    'output_dir':      'local_file/step3_file',
    'file_name':       'ALL',
    'arrow_scale':     150,          # 像素/度
    'faststart':       True,
    'field_map': {
        'px': 'track_x', 'py': 'track_y',
        'roll': 'angle_y', 'pitch': 'angle_x', 'yaw': 'tilt',
        'cam_pitch': 'g1'
    },
    'fov_h': 125.0, 'fov_v': 69.0,
    # -------- 新增面积阈值（像素^2） --------
    'min_track_area': 0.0,           # 小于该面积的 track 不写入 JPG/TXT
    'max_track_area': 9999999,       # 大于该面积的 track 不写入 JPG/TXT
}

# ----------------------- 数学 / 工具 ----------------------- #
def rot_matrix(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,      cp * sr,               cp * cr],
    ])

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)

def pixel_to_offsets(px, py, w, h, fov_h, fov_v):
    dx = (px - w / 2) / (w / 2)
    dy = (py - h / 2) / (h / 2)
    return dx * math.radians(fov_h) / 2, -dy * math.radians(fov_v) / 2

# ---- 统一重封装 H.264 + faststart ----
def reencode_faststart(src_tmp: str, dst: str, faststart: bool):
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error', '-i', src_tmp,
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
    ]
    if faststart:
        cmd += ['-movflags', '+faststart']
    cmd.append(dst)
    subprocess.run(cmd, check=True)
    os.remove(src_tmp)

# ----------------------- 主流程 ----------------------- #

def process_one(ts: str, cfg: dict):
    paths = {
        'vid': os.path.join(cfg['input_video_dir'], f'Cut_{ts}.mp4'),
        'trk': os.path.join(cfg['input_csv_dir'],  f'Msg_{ts}_Track.csv'),
        'raw': os.path.join(cfg['raw_csv_dir'],    f'Msg_{ts}.csv'),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        print(f'[WARN] Missing files for {ts}')
        return

    # Dataset/{images,labels}
    yolo_img_dir = os.path.join(cfg['output_dir'], 'dataset', 'images')
    yolo_lbl_dir = os.path.join(cfg['output_dir'], 'dataset', 'labels')
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_lbl_dir, exist_ok=True)

    fm = cfg['field_map']
    yaw0   = math.radians(pd.read_csv(paths['raw'], nrows=1)[fm['yaw']  ].iloc[0])
    pitch0 = math.radians(pd.read_csv(paths['raw'], nrows=1)[fm['pitch']].iloc[0])
    df = pd.read_csv(paths['trk'])

    cap = cv2.VideoCapture(paths['vid'])
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_mp4 = os.path.join(cfg['output_dir'], f'Observed_{ts}_tmp.mp4')
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    min_a = cfg['min_track_area']; max_a = cfg['max_track_area']
    scale = cfg['arrow_scale']

    for idx in tqdm(range(total), desc=ts, unit='f'):
        ret, frame = cap.read()
        if not ret:
            break
        frame_raw = frame.copy()  # 保留无可视化版本用于训练图像

        row = df.iloc[idx]
        if pd.isna(row[fm['px']]):
            writer.write(frame)
            continue

        # 像素、包围框尺寸 (px)
        px, py = int(row[fm['px']]), int(row[fm['py']])
        area   = max(row.get('track_area', 0.0), 1e-6)

        # 姿态角 (deg → rad)
        roll  = math.radians(row[fm['roll']])
        pitch = math.radians(row[fm['pitch']]) - pitch0
        yaw   = math.radians(row[fm['yaw']])   - yaw0

        # 机体法线向量
        n_body = rot_matrix(roll, -pitch, yaw) @ np.array([0, 0, 1])

        # 视向量
        off_yaw, off_pitch = pixel_to_offsets(px, py, w, h, cfg['fov_h'], cfg['fov_v'])
        cam_pitch = math.radians(row[fm['cam_pitch']])
        v_axis = unit(np.array([
            math.cos(cam_pitch + off_pitch) * math.cos(off_yaw),
            math.cos(cam_pitch + off_pitch) * math.sin(off_yaw),
            math.sin(cam_pitch + off_pitch),
        ]))

        # front / side angle（deg）
        front_angle = math.asin(np.dot(n_body, v_axis))
        horiz_dir   = np.array([math.sin(off_yaw), math.cos(off_yaw), 0.0])
        side_angle  = math.asin(np.dot(n_body, horiz_dir))
        
        l_box = 3.0 * math.sqrt(area)
        h_box = math.sqrt(area) * (1 + abs(math.sin(front_angle)))
        # 半尺寸
        hl, hh = l_box/2, h_box/2

        # 旋转矩阵（逆时针 front_angle）
        R = np.array([[ math.cos(-side_angle), -math.sin(-side_angle)],
                      [ math.sin(-side_angle),  math.cos(-side_angle)]])

        # 未旋转 4 点 (cx±hl, cy±hh)
        pts = np.array([[ hl,  hh], [-hl,  hh], [-hl, -hh], [ hl, -hh]], dtype=float)
        pts = (R @ pts.T).T + np.array([px, py])           # 旋转 + 平移
        pts_i = pts.astype(int)
        cv2.polylines(frame, [pts_i], True, (0,0,255), 2)

        # 箭头可视化
        dy_pix = scale * front_angle
        dx_pix = scale * side_angle
        cv2.arrowedLine(frame, (px - 20, py - 20), (px - 20, int(py - dy_pix - 20)),
                        (0, 0, 255), 2, tipLength=0.2)
        cv2.arrowedLine(frame, (px - 20, py - 20), (int(px - dx_pix - 20), py - 20),
                        (0, 165, 255), 2, tipLength=0.2)

        # ---------- JPG/TXT 保存（带面积过滤） ----------
        if min_a <= area <= max_a:

            pts_n = pts / np.array([w, h])
            flat = pts_n.reshape(-1) 
            
            label_line = "16 " + " ".join(f"{v:.6f}" for v in flat) + "\n"

            img_name = f"{ts}_{idx:06d}.jpg"
            lbl_name = img_name.replace('.jpg', '.txt')
            cv2.imwrite(os.path.join(yolo_img_dir, img_name), frame_raw)
            with open(os.path.join(yolo_lbl_dir, lbl_name), 'w') as f_lbl:
                f_lbl.write(label_line)
        # ------------------------------------------------

        writer.write(frame)

    cap.release()
    writer.release()

    # H.264 + faststart
    final_mp4 = os.path.join(cfg['output_dir'], f'Observed_{ts}.mp4')
    reencode_faststart(tmp_mp4, final_mp4, cfg['faststart'])
    print(f'[SAVE] {final_mp4}')

# --------------------- 配置 / 调度 --------------------- #

def load_cfg():
    here = os.path.dirname(__file__)
    cfg_p = os.path.join(here, '..', 'config.yaml')
    user_cfg = yaml.safe_load(open(cfg_p))\
        .get('step3_observed_angle', {}) if os.path.exists(cfg_p) else {}
    cfg = {**DEFAULT_CFG, **user_cfg}
    # 路径转绝对
    for k in ['raw_csv_dir', 'input_video_dir', 'input_csv_dir', 'output_dir']:
        cfg[k] = os.path.abspath(os.path.join(here, '..', cfg[k]))
    os.makedirs(cfg['output_dir'], exist_ok=True)
    return cfg

if __name__ == '__main__':
    cfg = load_cfg()
    if cfg['file_name'] == 'ALL':
        ts_list = [f[4:-4] for f in os.listdir(cfg['input_video_dir'])
                   if f.startswith('Cut_') and f.endswith('.mp4')]
    elif isinstance(cfg['file_name'], list):
        ts_list = [s.replace('Camera_', '').replace('.mp4', '') for s in cfg['file_name']]
    else:
        ts_list = [cfg['file_name'].replace('Camera_', '').replace('.mp4', '')]

    for ts in ts_list:
        process_one(ts, cfg)
