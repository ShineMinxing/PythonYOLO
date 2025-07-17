"""
step4_yolo_train.py • v2.3 (Pose+Detect with scale)
────────────────────────────────────────────────────────────
支持使用通用 `yolo12-pose.yaml` 自动下载并通过 `scale` 参数选择规模：
  e.g. scale='m'→YOLO12m, 's'→YOLO12s
同时微调检测框 + 1 个关键点 (x=front_angle, y=side_angle)

用法：
  python step4_yolo_train.py            # 读取 config.yaml
  python step4_yolo_train.py --scale s  # 改用 YOLO12s-pose

配置优先级：DEFAULTS → config.yaml → CLI
"""
from __future__ import annotations
import argparse, sys, yaml, random, urllib.request, shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] 请先安装 ultralytics: pip install ultralytics>=8.2.0")

# ------------------------ 默认配置 ------------------------ #
DEFAULTS = {
    'dataset_dir': 'local_file/step3_file/drone',
    'model':       'yolo12-pose.yaml',      # YAML → 自动下载后用于构建 Pose 模型
    'epochs':      10,
    'imgsz':       640,
    'batch':       16,
    'device':      'cuda:0',
    'freeze':      10,
    'project':     'local_file/step4_file',
    'name':        'drone_finetune',
    'resume':      False,
}
CFG_PATH = Path(__file__).resolve().parent.parent / 'config.yaml'
POSE_CFG_URL = (
    'https://raw.githubusercontent.com/ultralytics/ultralytics/main/' +
    'ultralytics/cfg/models/12/yolo12-pose.yaml'
)

# ------------------------ 工具函数 ------------------------ #

def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and k in base:
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml_cfg() -> dict:
    if CFG_PATH.exists():
        all_cfg = yaml.safe_load(open(CFG_PATH)) or {}
        return all_cfg.get('step4_yolo_train', {})
    return {}


def parse_cli() -> dict:
    ap = argparse.ArgumentParser('YOLO12-Pose fine-tune')
    for k, v in DEFAULTS.items():
        ap.add_argument(f'--{k}', type=type(v))
    args = ap.parse_args()
    return {k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}


def fetch_pose_yaml(path: Path) -> str:
    if path.suffix == '.yaml' and not path.exists():
        print(f"[INFO] 未找到 {path.name}，自动下载…")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(POSE_CFG_URL) as r, open(path, 'wb') as f:
            shutil.copyfileobj(r, f)
        print(f"[SAVE] {path}")
    return str(path)


def ensure_data_yaml(ds: Path) -> Path:
    y = ds / 'data.yaml'
    if not y.exists():
        yaml.safe_dump(
            {'train': 'images', 'val': 'images', 'nc': 1, 'names': ['drone'], 'kpt_shape': [1, 3]},
            open(y, 'w'),
        )
        print(f"[SAVE] {y}")
    return y

# -------------------------- 主函数 ------------------------- #

def main():
    # 合并配置
    cfg = deep_update(DEFAULTS.copy(), load_yaml_cfg())
    cfg = deep_update(cfg, parse_cli())

    # 解析 dataset_dir（项目根为基准）
    root = Path(__file__).resolve().parent.parent
    ds_dir = (root / cfg['dataset_dir']).resolve()
    assert ds_dir.exists(), f"dataset_dir 不存在: {ds_dir}"

    # 准备 data.yaml
    data_yaml = ensure_data_yaml(ds_dir)

    # 准备模型配置
    model_cfg = fetch_pose_yaml((root / cfg['model']).resolve())

    print(f"[INFO] task=pose  model={model_cfg}")
    model = YOLO(model_cfg)

    # 启动训练
    results = model.train(
        data=str(data_yaml),
        epochs=int(cfg['epochs']),
        imgsz=int(cfg['imgsz']),
        batch=int(cfg['batch']),
        device=cfg['device'],
        project=str((root / cfg['project']).resolve()),
        name=cfg['name'],
        resume=cfg['resume'],
        freeze=int(cfg['freeze']),
        cache=False,
        task='pose',
    )

    best = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"[DONE] 训练完成，最佳权重: {best}")
    print(f"ROS2 启动示例:\n  ros2 launch yolo_bringup yolo.launch.py model:={best}")

if __name__ == '__main__':
    random.seed(0)
    main()
