#!/usr/bin/env python3
"""
step4_yolo_train_local.py · v1.8 (fixed regex & last.pt support)
--------------------------------
* 增量训练：自动下载/复用 PT 权重 ➜ local_file/step4_file/models/
* 传入 YAML → 自动找同名 PT；传入 PT 或 best.pt/last.pt → 直接用作预训练权重
* 不再擅自 resume；若要接着训请显式 `--resume True`
* 禁用 AMP 自检 (ULTRALYTICS_NOAMP=1)，不会再偷偷拉 yolo11n.pt
"""
from __future__ import annotations
import argparse, random, shutil, subprocess, sys, os, re, yaml, requests, tqdm
from pathlib import Path

# ───────── Ultralytics 安装 & 导入 ─────────
try:
    from ultralytics import YOLO
    from ultralytics.data.split import autosplit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics>=8.3.0"])
    from ultralytics import YOLO
    from ultralytics.data.split import autosplit

# ───────── 默认参数 ─────────
DEFAULTS = {
    "dataset_dir": "local_file/step3_file/dataset",
    "model":       "yolo11m-obb.yaml",   # 可换 .yaml / .pt / best.pt / last.pt
    "epochs":      100,
    "imgsz":       640,
    "batch":       8,
    "device":      "cuda:0",
    "freeze":      10,
    "val":         True,
    "project":     "local_file/step4_file",
    "name":        "drone_finetune",
    "resume":      False,                # 手动开关
}

ROOT        = Path(__file__).resolve().parent.parent
CFG_PATH    = ROOT / "config.yaml"
MODELS_DIR  = ROOT / "local_file/step4_file/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ───────── 工具函数 ─────────
def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = deep_update(a[k], v) if isinstance(v, dict) and k in a else v
    return a


def load_yaml_cfg() -> dict:
    if CFG_PATH.exists():
        raw = yaml.safe_load(open(CFG_PATH, encoding="utf-8")) or {}
        return raw.get("step4_yolo_train", {})
    return {}


def cli_cfg() -> dict:
    ap = argparse.ArgumentParser("Fine-tune YOLO-OBB (local)")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v))
    return {k: v for k, v in vars(ap.parse_args()).items() if v is not None}


def http_download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] downloading {url}")
    r = requests.get(url, stream=True); r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dst, "wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192): f.write(chunk); bar.update(len(chunk))
    print("[SAVE]", dst)

# ───────── 保证 YAML / PT 存在 ─────────
def ensure_yaml(ver: str, scale: str) -> Path:
    generic = MODELS_DIR / f"yolo{ver}-obb.yaml"
    if not generic.exists():
        base = f"https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/{ver}/"
        http_download(base + generic.name, generic)
    if scale:
        target = MODELS_DIR / f"yolo{ver}{scale}-obb.yaml"
        if not target.exists(): shutil.copy(generic, target); print(f"[INFO] Copied {generic.name} → {target.name}")
        return target
    return generic


def ensure_pt(ver: str, scale: str) -> Path:
    pt_name = f"yolo{ver}{scale}-obb.pt" if scale else f"yolo{ver}-obb.pt"
    dest = MODELS_DIR / pt_name
    if dest.exists(): return dest
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{pt_name}"
    http_download(url, dest)
    return dest

# ───────── 数据准备 ─────────
def ensure_autosplit(ds: Path):
    if (ds/"autosplit_train.txt").exists() and (ds/"autosplit_val.txt").exists(): return
    print("[INFO] autosplitting train/val …")
    autosplit(path=str(ds/"images"), weights=(0.2,0.1,0.7), annotated_only=True)


def ensure_data_yaml(ds: Path) -> Path:
    yml = ds/"data.yaml"
    if not yml.exists(): yaml.safe_dump({"path":str(ds),"train":"autosplit_train.txt","val":"autosplit_val.txt","nc":1,"names":["drone"]}, open(yml,"w")); print("[SAVE]", yml)
    return yml

# ───────── 主流程 ─────────
def main():
    os.environ["ULTRALYTICS_NOAMP"] = "1"  # 不跑 AMP 自检
    cfg = deep_update(DEFAULTS.copy(), load_yaml_cfg())
    cfg = deep_update(cfg, cli_cfg())

    ds = (ROOT/cfg["dataset_dir"]).resolve()
    assert ds.exists(), f"dataset_dir 不存在: {ds}"
    ensure_autosplit(ds)
    data_yaml = ensure_data_yaml(ds)

    model_arg = Path(cfg["model"]).name.lower()
    # 支持 yolo<ver><scale>-obb.yaml/pt 和 best.pt/last.pt
    if model_arg in ("best.pt", "last.pt"):
        model_path = ROOT/cfg["model"]
        yaml_path  = "N/A (custom PT)"
        # 如果想从 best.pt 恢复训练，可传 --resume True
    else:
        m = re.fullmatch(r"yolo(\d+)([nslmx]?)-obb\.(yaml|pt)", model_arg)
        if not m: raise ValueError("--model 必须形如 yolo11m-obb.yaml / yolo12s-obb.pt / best.pt / last.pt")
        ver, scale, ext = m.groups(); scale = scale or ""
        yaml_path  = ensure_yaml(ver, scale)
        model_path = ensure_pt(ver, scale) if ext == "yaml" else ensure_pt(ver, scale)

    print("[INFO] YAML   :", yaml_path)
    print("[INFO] WEIGHTS:", model_path)

    trainer = YOLO(str(model_path), task="obb")
    trainer.train(
        data=str(data_yaml), epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]), device=cfg["device"], freeze=int(cfg["freeze"]),
        project=str((ROOT/cfg["project"]).resolve()), name=cfg["name"], resume=cfg["resume"],
        cache=False, val=cfg["val"], pretrained=False
    )

if __name__ == "__main__":
    random.seed(0)
    main()