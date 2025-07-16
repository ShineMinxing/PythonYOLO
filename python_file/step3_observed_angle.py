"""
step3_observed_angle.py  •  v2.1
-------------------------------------------------------------
为裁剪后视频中的无人机轨迹及其姿态绘制可视化箭头，并标注数值。

功能概览：
1. 从原始 IMU CSV 中读取无人机初始偏航（yaw0），用于归一化所有偏航角。
2. 读取插值后轨迹 CSV，获取每帧的像素坐标和机体姿态（roll, pitch, yaw）。
3. 结合相机内参（FOV）与像素偏移计算视线方向（v_axis）。
4. 计算无人机机体法线向量 n_body = R(roll, pitch, yaw) * [0,0,1]。
5. 将 n_body 分解为：
   • 平行于视线的分量（投影长度 proj_par）→ 竖直箭头表示，组成投影向量 n_parallel。
   • 垂直于视线且位于视平面的分量（n_perp）→ 水平箭头表示。
6. 箭头长度 ∝ 分量大小 × arrow_scale，便于可视化。
7. 在箭头旁标注对应数值（弧度），并生成最终带 faststart 的 MP4。

可配置项（见 YAML）：
- raw_csv_dir: 原始 IMU CSV 路径
- input_video_dir: 裁剪后 Cut_*.mp4 路径
- input_csv_dir: 插值轨迹 CSV 路径
- output_dir: 输出 Observed_*.mp4 路径
- file_name: 待处理文件名或 ALL
- arrow_scale: 箭头缩放系数（像素/弧度）
- faststart: 是否启用 MP4 faststart
- field_map: 各数据字段映射
- fov_h, fov_v: 相机水平/垂直视场角（度）
"""
import os, cv2, yaml, subprocess, pandas as pd, math, numpy as np
from tqdm import tqdm

# ---------------------------- 默认配置 ---------------------------- #
DEFAULT_CFG = {
    'raw_csv_dir': 'local_file/raw_file',
    'input_video_dir': 'local_file/step2_file',
    'input_csv_dir':  'local_file/step2_file',
    'output_dir':     'local_file/step3_file',
    'file_name':      'ALL',
    'arrow_scale':    150,
    'faststart':      True,
    'field_map': {
        'px': 'track_x', 'py': 'track_y',
        'roll':'angle_y','pitch':'angle_x','yaw':'tilt',
        'cam_pitch':'g1'
    },
    'fov_h': 125.0, 'fov_v': 69.0,
}

# ----------------------- 数学 / 工具 ----------------------- #
def rot_matrix(r,p,y):
    cr,sr = math.cos(r), math.sin(r)
    cp,sp = math.cos(p), math.sin(p)
    cy,sy = math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp   , cp*sr         , cp*cr        ]])

def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-8 else np.zeros_like(v)

def pixel_to_offsets(px,py,w,h,fov_h,fov_v):
    dx=(px-w/2)/(w/2); dy=(py-h/2)/(h/2)
    return dx*math.radians(fov_h)/2, -dy*math.radians(fov_v)/2

# ---- 统一重封装 H.264 + faststart ----
def reencode_faststart(src_tmp, dst, faststart):
    cmd=['ffmpeg','-y','-loglevel','error','-i',src_tmp,
         '-c:v','libx264','-preset','fast','-pix_fmt','yuv420p']
    if faststart: cmd+=['-movflags','+faststart']
    cmd.append(dst)
    subprocess.run(cmd, check=True)
    os.remove(src_tmp)

# ----------------------- 主流程 ----------------------- #
def process_one(ts, cfg):
    paths = {
        'vid': os.path.join(cfg['input_video_dir'], f'Cut_{ts}.mp4'),
        'trk': os.path.join(cfg['input_csv_dir'],  f'Msg_{ts}_Track.csv'),
        'raw': os.path.join(cfg['raw_csv_dir'],    f'Msg_{ts}.csv'),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        print(f'[WARN] Missing files for {ts}'); return

    fm=cfg['field_map']
    yaw0   = math.radians(pd.read_csv(paths['raw'], nrows=1)[fm['yaw'] ].iloc[0])
    pitch0 = math.radians(pd.read_csv(paths['raw'], nrows=1)[fm['pitch']].iloc[0])
    df     = pd.read_csv(paths['trk'])

    cap=cv2.VideoCapture(paths['vid'])
    fps=cap.get(cv2.CAP_PROP_FPS); w,h=int(cap.get(3)),int(cap.get(4))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_mp4=os.path.join(cfg['output_dir'], f'Observed_{ts}_tmp.mp4')
    writer=cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    scale=cfg['arrow_scale']
    rows_out=[]    # ← 追加输出 CSV 行

    for i in tqdm(range(total), desc=ts, unit='f'):
        ret, frame = cap.read();
        if not ret: break
        row = df.iloc[i]
        if pd.isna(row[fm['px']]):
            writer.write(frame); continue

        # 像素 → 画十字
        px, py = int(row[fm['px']]), int(row[fm['py']])
        cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)

        # 姿态角 (°→rad)
        roll  = math.radians(row[fm['roll']])
        pitch = math.radians(row[fm['pitch']]) - pitch0
        yaw   = math.radians(row[fm['yaw']]) - yaw0

        # 机体法线
        n_body = rot_matrix(roll, -pitch, yaw) @ np.array([0, 0, 1])

        # 视向量 v_axis
        off_yaw, off_pitch = pixel_to_offsets(px, py, w, h, cfg['fov_h'], cfg['fov_v'])
        cam_pitch = math.radians(row[fm['cam_pitch']])
        yaw_cam   = off_yaw                   # 右为正
        pitch_cam = cam_pitch + off_pitch     # 上为正
        v_axis = unit(np.array([
            math.cos(pitch_cam) * math.cos(yaw_cam),   # X
            math.cos(pitch_cam) * math.sin(yaw_cam),   # Y
            math.sin(pitch_cam),                       # Z
        ]))

        # 前后角度：无人机法线与视平面夹角
        dot_nv = np.dot(n_body, v_axis)
        front_angle = math.degrees(math.asin(dot_nv))

        # 左右角度：无人机法线与视垂面夹角
        horiz_dir = np.array([math.sin(off_yaw), math.cos(off_yaw), 0.0])
        dot_nd = np.dot(n_body, horiz_dir)
        side_angle = math.degrees(math.asin(dot_nd))

        # 像素箭头长度
        dy_pix = scale * front_angle
        dx_pix = scale * side_angle

        # 绘制箭头
        cv2.arrowedLine(frame, (px-20, py-20), (px-20, int(py - dy_pix-20)), (0, 0, 255),   2, tipLength=0.2)
        cv2.arrowedLine(frame, (px-20, py-20), (int(px - dx_pix-20), py-20), (0,165,255),   2, tipLength=0.2)

        # 文字标注：方向角、俯仰角、姿态角
        cv2.putText(frame, f"{math.degrees(off_yaw):+.3f} {math.degrees(off_pitch):+.3f} ", (px-40, py+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        cv2.putText(frame, f"R{math.degrees(roll):+.3f} P{math.degrees(pitch):+.3f} Y{math.degrees(yaw):+.3f}",(px-40, py+35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        cv2.putText(frame, f"Up{front_angle:+.3f} Left{side_angle:+.3f}", (px-40, py+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
        

        writer.write(frame)
        rows_out.append({'track_x':px,'track_y':py,'track_area':int(row['track_area']),
                         'front_angle':front_angle,'side_angle':side_angle})

    cap.release(); writer.release()

    # --- H.264 + faststart ---
    final_mp4=os.path.join(cfg['output_dir'], f'Observed_{ts}.mp4')
    reencode_faststart(tmp_mp4, final_mp4, cfg['faststart'])
    print(f'[SAVE] {final_mp4}')

    # --- Data_<ts>.csv ---
    df_out=pd.DataFrame(rows_out, columns=['track_x','track_y','track_area',
                                           'front_angle','side_angle'])
    csv_out=os.path.join(cfg['output_dir'], f'Data_{ts}.csv')
    df_out.to_csv(csv_out, index=False)
    print(f'[SAVE] {csv_out}')

# --------------------- 配置 / 调度 --------------------- #
def load_cfg():
    here=os.path.dirname(__file__)
    cfg_p=os.path.join(here,'..','config.yaml')
    user=yaml.safe_load(open(cfg_p))\
           .get('step3_observed_angle',{}) if os.path.exists(cfg_p) else {}
    cfg={**DEFAULT_CFG, **user}
    for k in ['raw_csv_dir','input_video_dir','input_csv_dir','output_dir']:
        cfg[k]=os.path.abspath(os.path.join(here,'..',cfg[k]))
    os.makedirs(cfg['output_dir'], exist_ok=True)
    return cfg

if __name__ == '__main__':
    cfg=load_cfg()
    if cfg['file_name']=='ALL':
        ts_list=[f[4:-4] for f in os.listdir(cfg['input_video_dir'])
                 if f.startswith('Cut_') and f.endswith('.mp4')]
    elif isinstance(cfg['file_name'],list):
        ts_list=[s.replace('Camera_','').replace('.mp4','') for s in cfg['file_name']]
    else:
        ts_list=[cfg['file_name'].replace('Camera_','').replace('.mp4','')]
    for ts in ts_list:
        process_one(ts, cfg)
