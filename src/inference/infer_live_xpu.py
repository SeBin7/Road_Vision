# src/inference/infer_live_xpu.py
# Real-time inference: CPU video decode + CPU OpenCV preproc + XPU normalize+CNN+GRU+MLP (+softmax)
# Comments are English-only per preference.

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Repo import
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.Mobilenet_hailo_4ch import MobileNetFeatureExtractor
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

# ---------------------------------------------------------------------
# Env flags
# ---------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "t", "yes", "y", "on")

CPU_ONLY = _env_bool("ROAD_VISION_CPU", False)
TIMING = _env_bool("ROAD_VISION_TIMING", False)
PREFETCH = _env_bool("ROAD_VISION_PREFETCH", True)          # overlap CPU preproc with XPU infer
PIN_MEMORY = _env_bool("ROAD_VISION_PIN_MEMORY", False)     # optional; may fail on some XPU setups
NOBLOCK = _env_bool("ROAD_VISION_NON_BLOCKING", True)       # optional; may fail depending on allocator

# ---------------------------------------------------------------------
# Model resolution (copied from your infer_batch_xpu.py pattern)
# ---------------------------------------------------------------------
MODEL_DIR_CANDIDATES = [
    ROOT / "models" / "models",
    ROOT / "models" / "server_model",
    ROOT / "models" / "4ch_results",
    ROOT,
]

CNN_WEIGHT_NAMES = [
    "cnn_feature_extractor_4ch_val.pth",
    "cnn_feature_extractor_4ch.pth",
    "best_cnn_feature_extractor_4ch.pth",
    "best_cnn_feature_extractor.pth",
]

CLS_WEIGHT_NAMES = [
    "gru_mlp_classifier_4ch_val.pth",
    "gru_mlp_classifier_4ch.pth",
    "best_gru_mlp_classifier_4ch.pth",
    "best_gru_mlp_classifier.pth",
]

LABEL_MAP = ["broken", "normal_road", "snow_road", "wet_road"]

def _find_model_file(names) -> Path:
    for name in names:
        for base in MODEL_DIR_CANDIDATES:
            p = base / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Cannot find any of {names} in {MODEL_DIR_CANDIDATES}")

class RoadVisionNet(torch.nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cnn = MobileNetFeatureExtractor()
        self.cls = GRU(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,4,224,224)
        feat = self.cnn(x)               # (N, feat_dim)
        seq = feat.unsqueeze(1)          # (N,1,feat_dim)
        logits = self.cls(seq)           # (N,num_classes)
        return logits

def get_device() -> torch.device:
    if CPU_ONLY:
        return torch.device("cpu")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def build_model(device: torch.device,
                cnn_weights: Optional[str] = None,
                cls_weights: Optional[str] = None) -> torch.nn.Module:
    cnn_path = Path(cnn_weights) if cnn_weights else _find_model_file(CNN_WEIGHT_NAMES)
    cls_path = Path(cls_weights) if cls_weights else _find_model_file(CLS_WEIGHT_NAMES)

    if TIMING:
        print(f"[model] cnn weights: {cnn_path}", flush=True)
        print(f"[model] cls weights: {cls_path}", flush=True)

    model = RoadVisionNet(num_classes=len(LABEL_MAP))
    model.cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
    model.cls.load_state_dict(torch.load(cls_path, map_location="cpu"))
    model.eval()
    model.to(device)

    # Warmup (important for XPU)
    with torch.inference_mode():
        x = torch.zeros((1, 4, 224, 224), device=device, dtype=torch.float32)
        _ = model(x)
        if device.type == "xpu":
            torch.xpu.synchronize()
    return model

# ---------------------------------------------------------------------
# Preprocess: CPU OpenCV -> uint8 HWC4, then normalize on device
# ---------------------------------------------------------------------
MEAN = torch.tensor([0.485, 0.456, 0.406, 0.5], dtype=torch.float32).view(1, 4, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225, 0.5], dtype=torch.float32).view(1, 4, 1, 1)

def preprocess_cpu_u8(frame_bgr: np.ndarray) -> np.ndarray:
    # Output: uint8 (224,224,4) in RGB+EDGE order
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.resize(edges, (224, 224), interpolation=cv2.INTER_NEAREST)

    img4 = np.dstack([rgb, edges])  # (224,224,4) uint8
    return img4

def to_device_normalized(img4_u8: np.ndarray,
                         device: torch.device,
                         mean_dev: torch.Tensor,
                         std_dev: torch.Tensor) -> torch.Tensor:
    # CPU uint8 -> device float32 -> normalize on device
    t = torch.from_numpy(img4_u8)                 # (H,W,4) uint8 on CPU
    t = t.permute(2, 0, 1).unsqueeze(0)           # (1,4,224,224) uint8 on CPU

    if PIN_MEMORY and device.type != "cpu":
        # Pinning can fail depending on XPU HostAllocator; keep it optional.
        try:
            t = t.pin_memory()
        except Exception:
            pass

    if device.type != "cpu":
        # non_blocking can fail if HostAllocator cannot allocate; keep fallback.
        try:
            t = t.to(device, non_blocking=NOBLOCK)
        except Exception:
            t = t.to(device)
    else:
        t = t.to(device)

    t = t.to(torch.float32).div_(255.0)          # float32 on device
    t = (t - mean_dev) / std_dev
    return t

# ---------------------------------------------------------------------
# Prefetch thread
# ---------------------------------------------------------------------
class Prefetcher:
    def __init__(self, cap: cv2.VideoCapture, maxsize: int = 2):
        self.cap = cap
        self.q: "queue.Queue[Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]]" = queue.Queue(maxsize=maxsize)
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.th.start()

    def _run(self) -> None:
        while not self.stop:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put((False, None, None))
                return
            img4_u8 = preprocess_cpu_u8(frame)
            self.q.put((True, frame, img4_u8))

    def get(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        return self.q.get()

    def close(self) -> None:
        self.stop = True

# ---------------------------------------------------------------------
# UI / overlay
# ---------------------------------------------------------------------
WINDOW = "Road-Vision (XPU Live)"

def draw_overlay(frame_bgr: np.ndarray, label: str, conf: float, fps_proc: float) -> None:
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, h / 1080.0)
    thickness = max(1, int(h / 540.0))

    cv2.putText(frame_bgr, f"{label}: {conf:.1f}%", (20, 50),
                font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Proc FPS: {fps_proc:.1f}", (20, 90),
                font, font_scale * 0.85, (220, 220, 220), thickness, cv2.LINE_AA)
    cv2.putText(frame_bgr, "Space: pause/play | Q: quit", (20, h - 20),
                font, font_scale * 0.7, (200, 200, 200), max(1, thickness - 1), cv2.LINE_AA)

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("--cnn-weights", type=str, default=None)
    ap.add_argument("--cls-weights", type=str, default=None)
    ap.add_argument("--no-prefetch", action="store_true")
    args = ap.parse_args()

    device = get_device()
    print(f"[device] using {device.type}", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    model = build_model(device, args.cnn_weights, args.cls_weights)

    mean_dev = MEAN.to(device)
    std_dev = STD.to(device)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    paused = False
    loop_fps = 0.0
    last_ts = time.perf_counter()

    use_prefetch = (PREFETCH and not args.no_prefetch)
    pf = Prefetcher(cap) if use_prefetch else None
    if pf:
        pf.start()

    print("❚❚ Space: pause/play | Q: quit", flush=True)

    while True:
        if paused:
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' '):
                paused = False
            elif key in (ord('q'), ord('Q')):
                break
            continue

        t0 = time.perf_counter()

        # Read + CPU preprocess
        tr0 = time.perf_counter()
        if pf:
            ok, frame, img4_u8 = pf.get()
            if not ok:
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            img4_u8 = preprocess_cpu_u8(frame)
        tr1 = time.perf_counter()

        # H2D + normalize + inference (XPU)
        tp0 = time.perf_counter()
        x = to_device_normalized(img4_u8, device, mean_dev, std_dev)
        tp1 = time.perf_counter()

        ti0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item() * 100.0)
        if device.type == "xpu":
            torch.xpu.synchronize()
        ti1 = time.perf_counter()

        # UI
        tu0 = time.perf_counter()
        now = time.perf_counter()
        dt = now - last_ts
        last_ts = now
        if dt > 0:
            loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / dt)

        draw_overlay(frame, LABEL_MAP[idx], conf, loop_fps)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        tu1 = time.perf_counter()

        if TIMING:
            read_ms = (tr1 - tr0) * 1000.0
            pre_ms = (tp1 - tp0) * 1000.0
            inf_ms = (ti1 - ti0) * 1000.0
            ui_ms = (tu1 - tu0) * 1000.0
            tot_ms = (tu1 - t0) * 1000.0
            fps = 1000.0 / tot_ms if tot_ms > 0 else 0.0
            print(f"[timing] read={read_ms:.2f}ms h2d+norm={pre_ms:.2f}ms infer={inf_ms:.2f}ms ui={ui_ms:.2f}ms total={tot_ms:.2f}ms fps≈{fps:.1f}",
                  flush=True)

        if key == ord(' '):
            paused = True
        elif key in (ord('q'), ord('Q')):
            break

    if pf:
        pf.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
