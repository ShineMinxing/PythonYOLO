import os
import cv2
import yaml
import subprocess
import pandas as pd
import math
from tqdm import tqdm

"""
step2_trace.py • v3.4 (fix CSV track assignment)
-------------------------------------------------------------
- 修正：IMU CSV 使用完整插值轨迹 traj 而非 traj_s
- 其他功能及日志保持不变
"""

# ---- FFmpeg Helpers ----

def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _remux(src, dst, faststart):
    print(f"[LOG] Muxing {src} -> {dst} (faststart={faststart})")
    cmd = [
        'ffmpeg','-y','-loglevel','error','-i', src,
        '-c:v','libx264','-preset','fast','-pix_fmt','yuv420p'
    ]
    if faststart:
        cmd += ['-movflags','+faststart']
    cmd.append(dst)
    _run(cmd)
    os.remove(src)


def _cut(src, dst, start_s, dur_s, faststart, out_fps=None):
    print(f"[LOG] Cutting {src}: start={start_s:.3f}s, duration={dur_s:.3f}s, fps_override={out_fps}, faststart={faststart}")
    tmp = dst.replace('.mp4', '_tmp.mp4')
    if out_fps and out_fps > 0:
        vf = f'fps={out_fps}'
        cmd = [
            'ffmpeg','-y','-loglevel','error',
            '-ss',f'{start_s:.3f}','-i',src,
            '-t',f'{dur_s:.3f}',
            '-vf',vf,
            '-c:v','libx264','-preset','fast','-pix_fmt','yuv420p'
        ]
    else:
        cmd = [
            'ffmpeg','-y','-loglevel','error',
            '-ss',f'{start_s:.3f}','-i',src,
            '-t',f'{dur_s:.3f}',
            '-c','copy'
        ]
    if faststart:
        cmd += ['-movflags','+faststart']
    cmd.append(tmp)
    _run(cmd)
    _remux(tmp, dst, faststart)

# ---- Tracking ----

def _pick_initial(df, idx, ref, cols, r0, growth):
    rx, ry = ref
    print(f"[LOG] Picking initial at row {idx}, ref=({rx},{ry}), r0={r0}, growth={growth}")
    if (rx, ry) != (0, 0):
        radius = r0
        while True:
            candidates = []
            for xc, yc, ac in cols:
                x = df.at[idx, xc]; y = df.at[idx, yc]; a = df.at[idx, ac]
                if pd.isna(x): continue
                if math.hypot(x-rx, y-ry) <= radius:
                    candidates.append((a, -math.hypot(x-rx, y-ry), x, y))
            if candidates:
                candidates.sort(reverse=True)
                a, _, x, y = candidates[0]
                print(f"[LOG] Initial found at (x={x}, y={y}, area={a}) with radius {radius}")
                return (x, y, a)
            radius += growth
    areas = [(df.at[idx, ac] if not pd.isna(df.at[idx, ac]) else 0) for _,_,ac in cols]
    k = int(pd.Series(areas).idxmax())
    xc, yc, ac = cols[k]
    x = df.at[idx, xc]; y = df.at[idx, yc]; a = df.at[idx, ac]
    print(f"[WARN] Fallback initial (x={x}, y={y}, area={a})")
    return x, y, a


def _track(df, s, e, r0, growth, ref):
    cols = [(f'x_{i}', f'y_{i}', f'area_{i}') for i in range(1,21)]
    px, py, pa = _pick_initial(df, s, ref, cols, r0, growth)
    radius = r0
    traj = []
    print(f"[LOG] Tracking frames {s} to {e}, initial point ({px},{py}), area={pa}")
    for i in tqdm(range(s, e+1), desc="Tracking", unit="frame"):
        row = df.iloc[i]
        candidates = []
        for xc, yc, ac in cols:
            x = row[xc]; y = row[yc]; a = row[ac]
            if pd.isna(x): continue
            d = math.hypot(x-px, y-py)
            if d <= radius:
                candidates.append((a, -d, x, y))
        if candidates:
            candidates.sort(reverse=True)
            a, _, x, y = candidates[0]
            px, py, pa = x, y, a
            radius = r0
            traj.append((x, y, a))
        else:
            radius += growth
            traj.append((math.nan, math.nan, math.nan))
    df_out = pd.DataFrame(traj, columns=['x','y','area'])
    df_out[['x','y']] = df_out[['x','y']].interpolate().round().astype('Int64')
    df_out['area'] = pd.to_numeric(df_out['area'], errors='coerce').ffill().round().astype('Int64')
    return df_out

# ---- Utils ----

def _first_valid(df, start):
    for i in range(start, len(df)):
        if df.iloc[i].filter(like='x_').notna().any(): return i
    return start


def _norm_ts(name):
    ts = name[:-4] if name.lower().endswith('.mp4') else name
    return ts[len('Camera_'):] if ts.startswith('Camera_') else ts

# ---- Main Pipeline ----

def _process(name, dirs, cfg):
    print(f"\n[PROCESS] File: {name}")
    ts = _norm_ts(name)
    vid = os.path.join(dirs['video'], f'Camera_{ts}.mp4')
    imu = os.path.join(dirs['video'], f'Msg_{ts}.csv')
    pts = os.path.join(dirs['points'], f'Trace_{ts}.csv')

    df_pts = pd.read_csv(pts)
    df_imu_full = pd.read_csv(imu)

    cap0 = cv2.VideoCapture(vid)
    src_fps = cap0.get(cv2.CAP_PROP_FPS)
    w = int(cap0.get(3)); h = int(cap0.get(4))
    total = int(cap0.get(7))
    cap0.release()
    print(f"[INFO] source fps={src_fps:.2f}, resolution={w}x{h}, total_frames={total}")

    target_fps = cfg.get('target_fps', 0)
    out_fps = target_fps if 0 < target_fps < src_fps else None
    stride = math.floor(src_fps / out_fps) if out_fps else 1
    print(f"[INFO] out_fps={out_fps}, stride={stride}")

    s0 = max(0, cfg.get('start_frame', 0) - 1)
    s = _first_valid(df_pts, s0)
    e = total - 1 if cfg.get('end_frame', 0) <= 0 else min(total - 1, cfg['end_frame'] - 1)
    print(f"[INFO] using frames {s}->{e}")

    ref = (cfg.get('ref_x', 0), cfg.get('ref_y', 0))
    traj = _track(df_pts, s, e, cfg['init_radius'], cfg['radius_growth'], ref)

    # CSV 裁剪 + 完整插值轨迹写入
    df_imu_cut = df_imu_full.iloc[s:e+1].reset_index(drop=True)
    for col in ['track_x', 'track_y', 'track_area']:
        if col not in df_imu_cut.columns:
            df_imu_cut[col] = pd.NA
    df_imu_cut[['track_x','track_y','track_area']] = traj.values
    out_dir = dirs['out']; os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, f'Msg_{ts}_Track.csv')
    df_imu_cut.to_csv(csv_out, index=False)
    print(f"[SAVE] {csv_out}")

    cut_out = os.path.join(out_dir, f'Cut_{ts}.mp4')
    _cut(vid, cut_out, s/src_fps, (e-s+1)/src_fps, cfg.get('faststart', True), out_fps)
    print(f"[SAVE] {cut_out}")

    tmp = os.path.join(out_dir, f'{ts}_tmp.mp4')
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), out_fps or src_fps, (w,h))
    cap = cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    frame_idx = s
    print(f"[RENDER] Target video with annotations")
    idxs = list(range(0, len(traj), stride)) if out_fps else list(range(len(traj)))
    traj_s = traj.iloc[idxs].reset_index(drop=True)
    for r in tqdm(traj_s.itertuples(), total=len(traj_s), desc="Rendering", unit="frame"):
        ret, frame = cap.read()
        if not ret: break
        x, y, a = r.x, r.y, r.area
        if not pd.isna(x):
            x, y, a = int(x), int(y), int(a)
            cv2.circle(frame, (x,y), 6, (0,0,255), -1)
            cv2.putText(frame, f"{x},{y},{a}", (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(frame, str(frame_idx), (w-60, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        writer.write(frame)
        for _ in range(stride-1): cap.grab()
        frame_idx += stride
    cap.release(); writer.release()
    target_out = os.path.join(out_dir, f'Target_{ts}.mp4')
    _remux(tmp, target_out, cfg.get('faststart', True))
    print(f"[SAVE] {target_out}\n")

if __name__ == '__main__':
    here = os.path.dirname(__file__)
    cfg = yaml.safe_load(open(os.path.join(here, '..', 'config.yaml')))['step2_trace']
    dirs = {
        'video': os.path.abspath(os.path.join(here, '..', cfg.get('input_video_dir','local_file/raw_file'))),
        'points':os.path.abspath(os.path.join(here, '..', cfg.get('input_csv_dir','local_file/step1_file'))),
        'out':   os.path.abspath(os.path.join(here, '..', cfg.get('output_dir','local_file/step2_file')))
    }
    sel = cfg.get('file_name', 'ALL')
    all_mp4 = [f for f in os.listdir(dirs['video']) if f.lower().endswith('.mp4')]
    targets = all_mp4 if sel=='ALL' else ([sel] if isinstance(sel,str) else sel)
    for v in targets:
        _process(v, dirs, cfg)
