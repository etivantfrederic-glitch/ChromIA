#!/usr/bin/env python3
"""
train_boxing.py

Version améliorée :
- Lit une config YAML (train_config.yaml)
- Adapte automatiquement imgsz et batch en fonction des images du data.yaml et de la VRAM disponible
- Logger console + fichier (project/name/logs/train.log)
- Optionnel: init Weights & Biases si activé
- Validation du data.yaml (train/val/nc/names) et échantillonnage d'images pour calculer la résolution moyenne

Usage:
  python train_boxing.py --config train_config.yaml
  python train_boxing.py --config train_config.yaml --override-device cuda:0

Dépendances :
  pip install ultralytics pillow pyyaml torch onnx onnxruntime

"""
import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import yaml

try:
    import torch
except Exception as e:
    print("Erreur: torch non installé. Installez via pip install torch")
    raise e

try:
    from ultralytics import YOLO
except Exception as e:
    print("Erreur: ultralytics non installé. Installez via pip install ultralytics")
    raise e

try:
    from PIL import Image
except Exception:
    Image = None  # we'll check at runtime


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_data_yaml(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        logging.error("data.yaml introuvable: %s", path)
        return False
    try:
        cfg = load_yaml(path)
    except Exception as e:
        logging.error("Impossible de lire %s: %s", path, e)
        return False
    for req in ("train", "val", "nc", "names"):
        if req not in cfg:
            logging.error("Champ requis manquant dans data.yaml: %s", req)
            return False
    logging.info("data.yaml valide (%s) - classes=%s", path, cfg.get("nc"))
    return True


def list_image_files(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    files = [f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    return files


def sample_image_sizes(paths: List[Path], sample_n: int = 200) -> Tuple[int, int]:
    """
    Ouvre jusqu'à sample_n images et retourne la médiane (width, height).
    """
    if not Image:
        raise RuntimeError("Pillow non installé. pip install pillow")
    if not paths:
        raise ValueError("Aucune image fournie pour l'échantillonnage")
    sample = paths.copy()
    random.shuffle(sample)
    sample = sample[:min(len(sample), sample_n)]
    widths = []
    heights = []
    for p in sample:
        try:
            with Image.open(p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            logging.debug("Impossible d'ouvrir %s", p)
    if not widths:
        raise RuntimeError("Aucune taille d'image récupérée (vérifiez les chemins et formats)")
    widths.sort()
    heights.sort()
    median_w = widths[len(widths) // 2]
    median_h = heights[len(heights) // 2]
    return median_w, median_h


def round_to_multiple_of_32(x: int) -> int:
    return int(math.ceil(x / 32) * 32)


def recommend_imgsz_from_data_yaml(data_yaml: str, sample_n: int = 200, min_imgsz: int = 640,
                                   max_imgsz: int = 1280) -> int:
    cfg = load_yaml(data_yaml)
    # gather train + val images
    imgs = []
    for key in ("train", "val", "test"):
        if key in cfg and cfg[key]:
            imgs += list_image_files(cfg[key])
    if not imgs:
        logging.warning("Aucun fichier image trouvé dans data.yaml paths; fallback à min_imgsz=%d", min_imgsz)
        return min_imgsz
    logging.info("Échantillonnage de %d images (max %d) pour estimer la résolution", min(len(imgs), sample_n), sample_n)
    mw, mh = sample_image_sizes(imgs, sample_n)
    chosen = max(mw, mh)
    # for small objects, prefer larger resolution: we take at least min_imgsz
    chosen = max(chosen, min_imgsz)
    chosen = min(chosen, max_imgsz)
    chosen = round_to_multiple_of_32(chosen)
    logging.info("Résolution médiane estimée: %dx%d -> imgsz recommandé: %d", mw, mh, chosen)
    return chosen


def get_gpu_memory_gb(device_str: Optional[str] = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    # device_str can be like 'cuda:0' or '0'
    if device_str is None:
        dev = 0
    else:
        try:
            if isinstance(device_str, str) and device_str.isdigit():
                dev = int(device_str)
            elif isinstance(device_str, str) and "cuda" in device_str:
                dev = int(device_str.split(":")[-1])
            else:
                dev = 0
        except Exception:
            dev = 0
    try:
        prop = torch.cuda.get_device_properties(dev)
        return prop.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def recommend_batch_from_vram(vram_gb: float, imgsz: int, base_batch_for_640: int = 8) -> int:
    """
    Heuristique:
      - define a base batch for imgsz=640 depending on vram
      - scale down batch when imgsz increases roughly by (imgsz/640)^2
    """
    if vram_gb <= 0:
        # cpu fallback
        return min(4, base_batch_for_640)
    if vram_gb >= 24:
        base = max(16, base_batch_for_640)
    elif vram_gb >= 16:
        base = 12
    elif vram_gb >= 12:
        base = 8
    elif vram_gb >= 8:
        base = 4
    elif vram_gb >= 4:
        base = 2
    else:
        base = 1
    scale = (imgsz / 640) ** 2
    batch = max(1, int(base / scale))
    return batch


def setup_logger(project: str, name: str) -> logging.Logger:
    out_dir = Path(project) / name / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"
    logger = logging.getLogger("train_boxing")
    logger.setLevel(logging.DEBUG)

    # avoid adding multiple handlers if called repeatedly
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(ch)

        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger


def build_train_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "data", "epochs", "batch", "imgsz", "device", "project", "name", "workers",
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "box", "cls", "obj",
        "patience", "augment", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
        "rect", "cache", "seed", "amp", "ema", "save", "save_period", "exist_ok",
        "plots", "verbose", "resume", "batch_size"
    }
    kwargs = {}
    for k, v in cfg.items():
        if k in allowed:
            if k == "batch_size":
                kwargs["batch"] = v
            else:
                kwargs[k] = v
    # ensure data is present
    if "data" not in kwargs and "data" in cfg:
        kwargs["data"] = cfg["data"]
    return kwargs


def find_weights(project: str, name: str) -> Optional[Path]:
    base = Path(project) / name / "weights"
    if base.exists():
        best = base / "best.pt"
        last = base / "last.pt"
        if best.exists():
            return best
        if last.exists():
            return last
        pts = sorted(list(base.glob("*.pt")), key=lambda p: p.stat().st_mtime)
        if pts:
            return pts[-1]
    candidates = [
        Path("runs") / "detect" / name / "weights" / "best.pt",
        Path("runs") / "detect" / name / "weights" / "last.pt",
        Path("runs") / "train" / name / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def export_weights(weights_path: str, exports: Dict[str, Any]) -> None:
    if not weights_path:
        logging.warning("Aucun poids pour export.")
        return
    y = YOLO(weights_path)
    fmt_list = exports.get("formats", ["onnx"])
    imgsz = exports.get("imgsz", 640)
    for fmt in fmt_list:
        try:
            logging.info("Export %s (imgsz=%s)...", fmt, imgsz)
            out = y.export(format=fmt, imgsz=imgsz)
            logging.info("Exporté %s => %s", fmt, out)
        except Exception as e:
            logging.error("Échec export %s: %s", fmt, e)


def main():
    parser = argparse.ArgumentParser(description="Entraînement YOLO (auto-tuning imgsz & batch) via config YAML")
    parser.add_argument("--config", "-c", type=str, default="train_config.yaml", help="Fichier config YAML")
    parser.add_argument("--override-device", type=str, default=None, help="Ex: 'cpu' or 'cuda:0' or '0'")
    parser.add_argument("--no-auto-tune", dest="auto_tune", action="store_false", help="Désactive l'auto-tuning imgsz/batch")
    args = parser.parse_args()

    # load config
    try:
        cfg = load_yaml(args.config)
    except Exception as e:
        print(f"Impossible de lire la config: {e}")
        sys.exit(1)

    data_yaml = cfg.get("data")
    if not data_yaml:
        print("La config doit contenir la clé 'data' pointant vers le data.yaml (Roboflow).")
        sys.exit(1)

    # set project/name defaults
    project = cfg.get("project", "runs/boxing")
    name = cfg.get("name", "boxing_exp")
    logger = setup_logger(project, name)
    # Replace global logging with our logger
    logging.getLogger().handlers = logger.handlers
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Chargement config: %s", args.config)
    if not validate_data_yaml(data_yaml):
        logging.error("Validation du data.yaml échouée.")
        sys.exit(1)

    # Device selection
    req_dev = args.override_device or cfg.get("device")
    # determine device
    if req_dev:
        if isinstance(req_dev, str) and req_dev.isdigit():
            req_dev = f"cuda:{req_dev}"
    if req_dev:
        if req_dev.lower() == "cpu":
            device = "cpu"
        elif "cuda" in str(req_dev).lower() and torch.cuda.is_available():
            device = req_dev
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logging.info("Device sélectionné: %s", device)

    # Auto-tune imgsz and batch
    auto_tune = cfg.get("auto_tune", True) and args.auto_tune
    if auto_tune:
        try:
            sample_n = int(cfg.get("sample_images", 200))
            min_imgsz = int(cfg.get("min_imgsz", 640))
            max_imgsz = int(cfg.get("max_imgsz", 1280))
            suggested_imgsz = recommend_imgsz_from_data_yaml(data_yaml, sample_n=sample_n,
                                                             min_imgsz=min_imgsz, max_imgsz=max_imgsz)
            # override cfg.imgsz only if not set by user explicitly
            if "imgsz" not in cfg or cfg.get("imgsz") in (None, ""):
                cfg["imgsz"] = suggested_imgsz
                logging.info("imgsz défini automatiquement à %d", suggested_imgsz)
            else:
                logging.info("imgsz fourni dans la config (%s) - auto-tune conserve la valeur fournie", cfg["imgsz"])
        except Exception as e:
            logging.warning("Auto-tune imgsz échoué: %s", e)

        # recommend batch based on vram
        vram = get_gpu_memory_gb(device)
        logging.info("VRAM détectée: %.2f GB", vram)
        try:
            imgsz_for_batch = int(cfg.get("imgsz", 640))
            suggested_batch = recommend_batch_from_vram(vram, imgsz_for_batch, base_batch_for_640=cfg.get("base_batch_for_640", 8))
            # only set if user didn't provide batch or batch_size
            if "batch" not in cfg and "batch_size" not in cfg:
                cfg["batch"] = suggested_batch
                logging.info("batch défini automatiquement à %d", suggested_batch)
            else:
                logging.info("batch fourni dans la config (%s) - auto-tune conserve la valeur fournie", cfg.get("batch") or cfg.get("batch_size"))
        except Exception as e:
            logging.warning("Auto-tune batch échoué: %s", e)
    else:
        logging.info("Auto-tune désactivé via option ou config.")

    # Model selection / mapping short names
    size_map = {"n": "yolov8n.pt", "s": "yolov8s.pt", "m": "yolov8m.pt", "l": "yolov8l.pt", "x": "yolov8x.pt"}
    model_spec = cfg.get("model", "n")
    if model_spec in size_map:
        model_spec = size_map[model_spec]
        logging.info("Mappage modèle court -> %s", model_spec)

    logging.info("Chargement du modèle: %s", model_spec)
    try:
        model = YOLO(model_spec)
    except Exception as e:
        logging.error("Échec chargement modèle %s: %s", model_spec, e)
        sys.exit(1)

    # build train kwargs and ensure presence of required keys
    train_kwargs = build_train_kwargs(cfg)
    train_kwargs.setdefault("data", data_yaml)
    train_kwargs.setdefault("epochs", int(cfg.get("epochs", 200)))
    if "imgsz" in cfg:
        train_kwargs["imgsz"] = cfg["imgsz"]
    # ensure device passed
    train_kwargs["device"] = device

    # print summary
    logging.info("Résumé entrainement:")
    logging.info("  model: %s", model_spec)
    for k in ("data", "epochs", "imgsz", "batch", "device", "project", "name"):
        if k in train_kwargs:
            logging.info("  %s: %s", k, train_kwargs[k])

    # Initialize Weights & Biases if requested
    if cfg.get("use_wandb", False):
        try:
            import wandb
            wandb_mode = cfg.get("wandb", {}).get("mode", "online")
            wandb_proj = cfg.get("wandb", {}).get("project", Path(project).name)
            wandb_run_name = cfg.get("wandb", {}).get("name", train_kwargs.get("name", "run"))
            wandb.init(project=wandb_proj, name=wandb_run_name, mode=wandb_mode)
            logging.info("W&B initialisé (project=%s, name=%s)", wandb_proj, wandb_run_name)
        except Exception as e:
            logging.warning("W&B demandé mais échec d'initialisation: %s", e)

    # Launch training
    try:
        logging.info("Lancement de l'entraînement...")
        model.train(**train_kwargs)
        logging.info("Entraînement terminé.")
    except Exception as e:
        logging.exception("Erreur lors de l'entraînement: %s", e)
        sys.exit(1)

    # Find weights
    project_out = train_kwargs.get("project", project)
    name_out = train_kwargs.get("name", name)
    best_path = find_weights(project_out, name_out)
    if best_path:
        logging.info("Poids trouvés: %s", best_path)
        best_path = str(best_path)
    else:
        logging.warning("Aucun poids best.pt/last.pt trouvé automatiquement.")

    # Export if requested
    if "export" in cfg:
        logging.info("Export demandé dans la config.")
        export_weights(best_path, cfg.get("export", {}))

    # Evaluate if requested
    if cfg.get("evaluate", False):
        eval_imgsz = int(cfg.get("eval_imgsz", cfg.get("imgsz", 640)))
        eval_batch = int(cfg.get("eval_batch", max(1, int(train_kwargs.get("batch", 4)))))
        if best_path:
            logging.info("Évaluation du modèle %s", best_path)
            try:
                model_eval = YOLO(best_path)
                res = model_eval.val(data=data_yaml, imgsz=eval_imgsz, batch=eval_batch, save_json=True)
                logging.info("Évaluation terminée: %s", getattr(res, "box", "N/A"))
            except Exception as e:
                logging.error("Erreur lors de l'évaluation: %s", e)
        else:
            logging.warning("Pas de poids disponibles pour évaluation.")

    logging.info("Terminé. Logs: %s/%s/logs/train.log", project_out, name_out)


if __name__ == "__main__":
    main()