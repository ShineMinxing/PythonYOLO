"""
step4_yolo_train_local.py · v1.6
--------------------------------
* **增量训练**：无论传入 `.yaml` 还是 `.pt`，都会下载对应 **预训练权重**
  到 `/home/unitree/WorkSpace/PythonYOLO/local_file/step4_file/models/`
  并在此基础上微调。
* **AMP 自检禁用**：设置 `ULTRALYTICS_NOAMP=1`，避免再拉取 `yolo11n.pt`。
* **支持 YAML→PT 转换**：如给 `yolo11m-obb.yaml`，脚本自动下载
  `yolo11m-obb.pt` 并以此权重训练（`pretrained=False`）。
"""

from __future__ import annotations
import argparse, random, shutil, subprocess, sys, os, re, urllib.request, yaml, requests, tqdm
from pathlib import Path

# ───────── Ultralytics 安装 ─────────
try:
    from ultralytics import YOLO
    from ultralytics.data.split import autosplit
except ImportError:                               # 首次运行自动安装
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics>=8.3.0"])
    from ultralytics import YOLO
    from ultralytics.data.split import autosplit

# ───────── 默认参数 ─────────
DEFAULTS = {
    "dataset_dir": "local_file/step3_file/dataset",
    "model":       "yolo11m-obb.yaml",   # .yaml 或 .pt 均可
    "epochs":      100,
    "imgsz":       640,
    "batch":       8,
    "device":      "cuda:0",
    "freeze":      10,
    "val":         True,
    "project":     "local_file/step4_file",
    "name":        "drone_finetune",
    "resume":      False,
}

ROOT        = Path(__file__).resolve().parent.parent
CFG_FILE    = ROOT / "config.yaml"
MODELS_DIR  = ROOT / "local_file/step4_file/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ───────── 辅助函数 ─────────
def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = deep_update(a[k], v) if isinstance(v, dict) and k in a else v
    return a

def load_cfg_yaml() -> dict:
    if CFG_FILE.exists():
        full = yaml.safe_load(open(CFG_FILE, encoding="utf-8")) or {}
        return full.get("step4_yolo_train", {})
    return {}

def cli_args() -> dict:
    ap = argparse.ArgumentParser("Fine-tune YOLO-OBB (local)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v))
    return {k: v for k, v in vars(ap.parse_args()).items() if v is not None}

def http_download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] downloading {url}")
    r = requests.get(url, stream=True); r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dst, "wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk); bar.update(len(chunk))
    print("[SAVE]", dst)

# ───────── 确保 YAML/PT 存在 ─────────
def ensure_yaml(ver: str, scale: str) -> Path:
    """下载 yolo<ver>-obb.yaml，并对需要的缩放版做拷贝。"""
    generic = MODELS_DIR / f"yolo{ver}-obb.yaml"
    if not generic.exists():
        base = f"https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/{ver}/"
        http_download(base + generic.name, generic)
    if scale:                                       # 需 m/s/l/x 缩放版
        target = MODELS_DIR / f"yolo{ver}{scale}-obb.yaml"
        if not target.exists():
            shutil.copy(generic, target)
            print(f"[INFO] Copied {generic.name} → {target.name}")
        return target
    return generic

def ensure_pt(ver: str, scale: str) -> Path:
    """下载 yolo<ver><scale>-obb.pt 预训练权重。"""
    pt_name = f"yolo{ver}{scale}-obb.pt" if scale else f"yolo{ver}-obb.pt"
    dest = MODELS_DIR / pt_name
    if dest.exists():
        return dest
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{pt_name}"
    http_download(url, dest)
    return dest

# ───────── 数据准备 ─────────
def ensure_autosplit(ds: Path):
    t, v = ds/"autosplit_train.txt", ds/"autosplit_val.txt"
    if t.exists() and v.exists():
        return
    print("[INFO] autosplitting train/val …")
    img_dir = ds/"images"
    assert img_dir.exists(), f"{img_dir} 不存在"
    autosplit(path=str(img_dir), weights=(0.8,0.2,0.0), annotated_only=True)

def ensure_data_yaml(ds: Path) -> Path:
    yml = ds/"data.yaml"
    if not yml.exists():
        yaml.safe_dump({
            "path": str(ds),
            "train": "autosplit_train.txt",
            "val":   "autosplit_val.txt",
            "nc":    1,
            "names": ["drone"],
        }, open(yml, "w"))
        print("[SAVE]", yml)
    return yml

# ───────── 主入口 ─────────
def main():
    os.environ["ULTRALYTICS_NOAMP"] = "1"      # 禁掉 AMP 自检 → 不再下载 11n.pt
    cfg = deep_update(DEFAULTS.copy(), load_cfg_yaml())
    cfg = deep_update(cfg, cli_args())

    # 数据集处理
    ds = (ROOT / cfg["dataset_dir"]).resolve()
    assert ds.exists(), f"dataset_dir 不存在: {ds}"
    ensure_autosplit(ds)
    data_yaml = ensure_data_yaml(ds)

    # 解析模型参数
    model_arg = Path(cfg["model"]).name.lower()
    m = re.fullmatch(r"yolo(\d+)([nslmx]?)-obb\.(yaml|pt)", model_arg)
    if not m:
        raise ValueError("--model 必须形如 yolo11m-obb.yaml / yolo12s-obb.pt")
    ver, scale, ext = m.groups()
    scale = scale or ""                         # '' 代表 n 版本

    # 保证 YAML 与 PT 均存在
    yaml_path = ensure_yaml(ver, scale)
    pt_path   = ensure_pt(ver, scale)

    print("[INFO] YAML   :", yaml_path)
    print("[INFO] WEIGHTS:", pt_path)

    # 启动微调
    YOLO(pt_path, task="obb").train(
        data      = str(data_yaml),
        epochs    = int(cfg["epochs"]),
        imgsz     = int(cfg["imgsz"]),
        batch     = int(cfg["batch"]),
        device    = cfg["device"],
        project   = str((ROOT / cfg["project"]).resolve()),
        name      = cfg["name"],
        resume    = cfg["resume"],
        freeze    = int(cfg["freeze"]),
        cache     = False,
        val       = cfg["val"],
        pretrained=False                      # 我们已显式提供权重
    )

if __name__ == "__main__":
    random.seed(0)
    main()
