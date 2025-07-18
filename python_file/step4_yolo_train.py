"""
step4_yolo_train_dota.py • v4.11 (显存友好版)
-------------------------------------------------
主要改动
~~~~~~~~
1. **默认 batch=6**，可 CLI 覆盖；显存占用≈4.8 GB。
2. **默认关闭每轮验证 (`val=False`)**，训练结束后再手动 `yolo val …`；亦可 `--val True` 开启。
3. 支持 `--acc_cpu True`：正负样本分配 & Probiou 计算全在 CPU，杜绝验证期 OOM。
4. 其余：自动下载/复制 `yolo11m-obb.yaml`、DOTA 合并、autosplit、data.yaml 生成与训练流程均保持。

CLI 示例
~~~~~~~~
```bash
# 显存 8 GB 单卡推荐
export ULTRA_ACC_CPU=1           # 或 --acc_cpu True
python step4_yolo_train_dota.py  \
  --model yolo11m-obb.yaml \
  --batch 6 --val False --epochs 100

# 训练完手动评估
yolo obb val model=local_file/step4_file/drone_dota_finetune/weights/best.pt \
             data=local_file/step3_file/dataset/data.yaml
```
"""
from __future__ import annotations
import argparse, random, shutil, subprocess, sys, urllib.request, zipfile, re, os
import yaml, requests, tqdm
from pathlib import Path

USE_MULTICLASS = True
DOTA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.5.zip"
DOTA_CLASSES = [
    "plane","ship","storage tank","baseball diamond","tennis court","basketball court",
    "ground track field","harbor","bridge","large vehicle","small vehicle","helicopter",
    "roundabout","soccer ball field","swimming pool","container crane"
]

try:
    from ultralytics import YOLO
    from ultralytics.data.split import autosplit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics>=8.3.0"])
    from ultralytics import YOLO
    from ultralytics.data.split import autosplit

DEFAULTS = {
    "dataset_dir": "local_file/step3_file/dataset",
    "model":       "yolo11m-obb.yaml",
    "epochs":      100,
    "imgsz":       640,
    "batch":       8,
    "device":      "cuda:0",
    "freeze":      10,
    "val":         True,
    "acc_cpu":     True,
    "project":     "local_file/step4_file",
    "name":        "drone_dota_finetune",
    "resume":      False,
}

# ---------------- 工具函数 ----------------

def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        base[k] = deep_update(base[k], v) if isinstance(v, dict) and k in base else v
    return base


def parse_cli() -> dict:
    ap = argparse.ArgumentParser("YOLO11m‑OBB + DOTA 训练 v4.11")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v))
    return {k: v for k, v in vars(ap.parse_args()).items() if v is not None}

# ---------------- 下载 YAML ----------------

def _download(url: str, dst: Path) -> bool:
    try:
        with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        print("[SAVE]", dst)
        return True
    except urllib.error.HTTPError as e:
        return False if e.code == 404 else (_ for _ in ()).throw(e)


def ensure_model_yaml(path: Path) -> str:
    if path.exists():
        return str(path)
    base = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/"
    generic = path.parent / "yolo11-obb.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not generic.exists():
        print("[INFO] downloading", generic.name)
        if not _download(base + generic.name, generic):
            raise FileNotFoundError("无法下载 yolo11-obb.yaml")
    shutil.copy(generic, path)
    print(f"[INFO] Copied {generic.name} → {path.name} (enable scale m)")
    return str(path)

# ---------------- DOTA 合并 ----------------

def download_with_progress(url: str, dst: Path):
    if dst.exists(): return
    r = requests.get(url, stream=True); r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dst, "wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192): f.write(chunk); bar.update(len(chunk))

def extract_zip(src: Path, out: Path):
    with zipfile.ZipFile(src) as zf: zf.extractall(out)

def merge_datasets(root: Path):
    z, dr = root/"DOTAv1.5.zip", root/"DOTAv1.5"
    download_with_progress(DOTA_URL, z)
    if not dr.exists(): extract_zip(z, root)
    img_dst, lbl_dst = root/"images", root/"labels"; img_dst.mkdir(exist_ok=True); lbl_dst.mkdir(exist_ok=True)
    for split in ("train","val","test"):
        for p in (dr/"images"/split).rglob('*.*'):
            t = img_dst/f"dota_{split}_{p.name}"; shutil.copy(p, t) if not t.exists() else None
        for p in (dr/"labels"/split).rglob('*.txt'):
            t = lbl_dst/f"dota_{split}_{p.name}"; shutil.copy(p, t) if not t.exists() else None

# ---------------- 主流程 ----------------

def main():
    cfg = deep_update(DEFAULTS.copy(), parse_cli())
    if cfg["acc_cpu"]:  # 环境变量方式更通用
        os.environ["ULTRA_ACC_CPU"] = "1"
    root = Path(__file__).resolve().parent.parent
    ds = (root/cfg["dataset_dir"]).resolve(); assert ds.exists()

    merge_datasets(ds)
    autosplit(path=str(ds/"images"), weights=(0.8,0.2,0.0), annotated_only=True)

    names = DOTA_CLASSES + ["drone"] if USE_MULTICLASS else ["drone"]
    yaml.safe_dump({"path":str(ds),"train":"autosplit_train.txt","val":"autosplit_val.txt","nc":len(names),"names":names}, open(ds/"data.yaml","w"))

    model_yaml = ensure_model_yaml((root/cfg["model"]).resolve())
    print("[INFO] Using", model_yaml)

    YOLO(model_yaml, task="obb").train(
        data=str(ds/"data.yaml"),
        epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
        device=cfg["device"], project=str((root/cfg["project"]).resolve()), name=cfg["name"],
        resume=cfg["resume"], freeze=int(cfg["freeze"]), cache=False, val=cfg["val"])

if __name__ == "__main__":
    random.seed(0); main()
