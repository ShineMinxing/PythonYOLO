import os
import yaml
import cv2
import subprocess
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

"""
step1_moving_points.py • v2.1
-------------------------------------------------------------
• CSV 输出改为保存像素坐标，不再读取或计算视场角 fov_h/fov_v。
• 新增像素列 x_i, y_i 替代 angle_h_i, angle_v_i。
• 其它功能保持不变。
"""

# 颜色列表
COLORS=[
    (0,0,255),(0,255,255),(255,0,0),(0,255,0),(255,0,255),
    (255,255,0),(0,128,255),(128,0,255),(255,128,0),(128,255,0),
    (128,128,255),(255,64,64),(64,255,64),(64,64,255),(255,255,128),
    (128,255,255),(255,128,255),(192,64,64),(64,192,64),(64,64,192)
]

# ------------ FFmpeg helpers ------------

def run_ff(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def open_writer(path,fps,size):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size), 'mp4v'


def remux(tmp, outp, faststart, reencode):
    cmd=['ffmpeg','-y','-loglevel','error','-i',tmp]
    cmd+=['-c:v','libx264','-preset','fast','-pix_fmt','yuv420p'] if reencode else ['-c','copy']
    if faststart: cmd+=['-movflags','+faststart']
    cmd.append(outp)
    run_ff(cmd)
    os.remove(tmp)

# ---------- Contour merging ------------

def merge_contours(cnts, thr):
    if len(cnts)<=1: return cnts
    boxes=[cv2.boundingRect(c) for c in cnts]
    merged=[False]*len(cnts)
    out=[]
    for i in range(len(cnts)):
        if merged[i]: continue
        group=[i]
        xi,yi,wi,hi=boxes[i]
        cxi,cyi=xi+wi/2, yi+hi/2
        for j in range(i+1,len(cnts)):
            if merged[j]: continue
            xj,yj,wj,hj=boxes[j]; cxj,cyj=xj+wj/2, yj+hj/2
            if math.hypot(cxi-cxj, cyi-cyj) < thr:
                merged[j]=True; group.append(j)
        pts = np.concatenate([cnts[k] for k in group])
        out.append(cv2.convexHull(pts))
    return out

# ---------------- Core -----------------

def process_one(video_path, csv_path, out_video, out_csv, cfg):
    # 参数
    max_pts     = int(cfg.get('max_points', 20))
    min_area    = float(cfg.get('min_area', 20))
    max_ar      = float(cfg.get('max_area_ratio', 0.1))
    faststart   = bool(cfg.get('faststart', True))
    reenc       = bool(cfg.get('reencode', False))

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('[WARN] Cannot open', video_path)
        return
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 临时写入
    tmp, codec = open_writer(out_video.replace('.mp4','_tmp.mp4'), fps, (w,h))
    auto_reenc = (codec=='mp4v' and not reenc)
    do_reenc   = reenc or auto_reenc
    print(f'[LOG] Codec={codec}, reencode={do_reenc}')

    # 背景与形态学
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=False)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    frame_area = w*h

    rows=[]; tot_targets=0
    for idx in tqdm(range(total), desc=os.path.basename(video_path)):
        cap=cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read(); cap.release()
        if not ret: break

        mask = bg.apply(frame)
        _,mask = cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, ker, iterations=2)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in cnts if min_area <= cv2.contourArea(c) <= max_ar*frame_area]
        cnts = merge_contours(filtered, thr=int(min(w,h)*0.05))
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_pts]
        tot_targets += len(cnts)

        row={}
        for i,c in enumerate(cnts):
            M=cv2.moments(c)
            if M['m00']==0: continue
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            r = max(1, int(math.sqrt(cv2.contourArea(c)/math.pi)))
            cv2.circle(frame,(cx,cy),r,COLORS[i%len(COLORS)],-1)
            row[f'x_{i+1}'] = cx
            row[f'y_{i+1}'] = cy
            row[f'area_{i+1}'] = cv2.contourArea(c)
        rows.append(row)
        tmp.write(frame) if isinstance(tmp, cv2.VideoWriter) else None
    if isinstance(tmp, cv2.VideoWriter): tmp.release()

    # 保存 CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'[SAVE] CSV: {out_csv}')

    # 转封装/重编码
    tmp_path = out_video.replace('.mp4','_tmp.mp4')
    remux(tmp_path, out_video, faststart, do_reenc)
    print(f'[SAVE] Video: {out_video} | avg_targets/frame={tot_targets/len(rows) if rows else 0:.2f}')

# -------------- Entry ------------------

def main():
    here=os.path.dirname(__file__)
    cfg_all=yaml.safe_load(open(os.path.join(here,'..','config.yaml')))
    cfg=cfg_all.get('step1_moving_points',{})
    in_dir = os.path.abspath(os.path.join(here,'..',cfg.get('input_dir','local_file/raw_file')))
    out_dir= os.path.abspath(os.path.join(here,'..',cfg.get('output_dir','local_file/step1_file')))
    os.makedirs(out_dir, exist_ok=True)

    sel = cfg.get('file_name','ALL')
    vids = ([f for f in os.listdir(in_dir) if f.lower().endswith('.mp4')] 
            if sel=='ALL' else (sel if isinstance(sel,list) else [sel]))

    for v in vids:
        ts = v.replace('Camera_','').replace('.mp4','')
        process_one(
            os.path.join(in_dir,v),
            os.path.join(in_dir,f'Msg_{ts}.csv'),
            os.path.join(out_dir,v.replace('.mp4','_Trace.mp4')),
            os.path.join(out_dir,f'Trace_{ts}.csv'),
            cfg
        )

if __name__=='__main__':
    main()
