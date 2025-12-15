# src/inference/infer_batch_xpu.py
# Offline batch inference: load full video, preprocess to 4ch, run CNN+GRU on XPU/CPU,
# then replay video with overlayed predictions.

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ----------------------------------------------------------------------
# Project path setup (similar to infer_4k.py)
# ----------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.Mobilenet_hailo_4ch import MobileNetFeatureExtractor
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

# Optional IPEX
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    ipex = None
    IPEX_AVAILABLE = False

# ----------------------------------------------------------------------
# Global config
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

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

label_map = {0: "broken", 1: "normal_road", 2: "snow_road", 3: "wet_road"}

FORCE_CPU = os.environ.get("ROAD_VISION_CPU", "").lower() in {"1", "true", "yes"}
DEBUG     = os.environ.get("ROAD_VISION_DEBUG", "").lower() in {"1", "true", "yes"}
TIMING    = DEBUG or os.environ.get("ROAD_VISION_TIMING", "").lower() in {"1", "true", "yes"}
TIMING_INT = int(os.environ.get("ROAD_VISION_TIMING_INT", "30"))

# UI/look constants (align with infer_4k)
SEQ_LEN      = 5
WINDOW_NAME  = "Road-Vision-Live"
BAR_COLOR    = (0, 255, 0)  # BGR
FONT         = cv2.FONT_HERSHEY_SIMPLEX
MAX_SCREEN_H = 1080


def _dbg(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)

def get_text_params(h: int):
    """프레임 높이에 비례한 (폰트스케일, 두께, 위치 y오프셋) 반환"""
    font_scale = h / 1080 * 1.0          # 1080p→1.0, 4K(2160p)→2.0
    thickness  = max(1, int(h / 1080 * 2))
    y_pred     = int(h * 0.04)           # 상태바 y위치
    y_time     = int(h * 0.08)           # 시간바 y위치
    return font_scale, thickness, y_pred, y_time


# ----------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------
def select_device() -> torch.device:
    if (not FORCE_CPU) and torch.xpu.is_available():
        dev = torch.device("xpu")
    else:
        dev = torch.device("cpu")
    if DEBUG or TIMING:
        print(f"[device] using {dev}", flush=True)
    return dev


# ----------------------------------------------------------------------
# Model weight resolution helper
# ----------------------------------------------------------------------
def _find_model_file(names) -> Path:
    """Return the first existing path among expected model dirs and names."""
    for name in names:
        for base in MODEL_DIR_CANDIDATES:
            candidate = base / name
            if candidate.exists():
                return candidate
    return ROOT / names[0]


# ----------------------------------------------------------------------
# Full model: CNN + GRU + MLP
# ----------------------------------------------------------------------
class RoadVisionNet(nn.Module):
    """
    CNN feature extractor + GRU+MLP classifier.

    Input:
        x: (N, 4, 224, 224)
    Output:
        logits: (N, num_classes)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cnn = MobileNetFeatureExtractor()
        self.cls = GRU(feature_dim=128)  # same constructor as infer_4k.py
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)              # (N, 128)
        seq = feats.unsqueeze(1)         # (N, 1, 128)
        logits = self.cls(seq)           # (N, num_classes)
        return logits


def build_model(device: torch.device) -> nn.Module:
    """Create RoadVisionNet, load CNN/CLS weights, move to device, and optimize."""
    cnn_path = _find_model_file(CNN_WEIGHT_NAMES)
    cls_path = _find_model_file(CLS_WEIGHT_NAMES)

    _dbg(f"[model] cnn weights: {cnn_path}")
    _dbg(f"[model] cls weights: {cls_path}")

    model = RoadVisionNet(num_classes=len(label_map))
    cnn_state = torch.load(cnn_path, map_location="cpu")
    cls_state = torch.load(cls_path, map_location="cpu")
    model.cnn.load_state_dict(cnn_state)
    model.cls.load_state_dict(cls_state)

    model = model.to(device)
    model.eval()

    if device.type == "xpu" and IPEX_AVAILABLE:
        _dbg("[model] applying IPEX optimization for XPU")
        model = ipex.optimize(model, dtype=torch.float32, inplace=True)

    return model


# ----------------------------------------------------------------------
# Frame -> (4,224,224) tensor preprocessor
# ----------------------------------------------------------------------
class FourChannelPreprocessor:
    """Convert BGR frame to normalized 4-channel tensor (RGB + edge)."""

    def __init__(self):
        self.transform_rgb = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(4, 1, 1)

    def __call__(self, frame_bgr) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t_rgb = self.transform_rgb(pil)  # (3,224,224)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.resize(edges, (224, 224))
        t_edge = torch.from_numpy(edges).unsqueeze(0).float().div(255.0)  # (1,224,224)

        img4 = torch.cat([t_rgb, t_edge], dim=0)  # (4,224,224)
        img4 = (img4 - self.mean) / self.std
        return img4




class FourChannelPreprocessorFast:
    """
    Faster 4ch preprocessor:
      - resize FIRST (224x224) to reduce CPU work
      - compute edge on resized grayscale
      - avoid PIL/torchvision overhead
      - use NumPy vectorization; convert once to torch (CHW float32)
    """
    def __init__(
        self,
        out_hw: tuple[int, int] = (224, 224),
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> None:
        self.out_hw = out_hw  # (H, W)
        self.canny_low = canny_low
        self.canny_high = canny_high

        # Match legacy normalization (RGB Imagenet + edge in [0,1] normalized to [-1,1])
        self.mean_np = np.array([0.485, 0.456, 0.406, 0.5], dtype=np.float32)
        self.inv_std_np = (1.0 / np.array([0.229, 0.224, 0.225, 0.5], dtype=np.float32)).astype(np.float32)
        self.bias_np = (-self.mean_np * self.inv_std_np).astype(np.float32)

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # Resize early (significant speedup vs full-res edge+resize)
        h, w = self.out_hw
        frame_small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # BGR -> RGB in 224x224
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        # Edge on resized grayscale
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, self.canny_low, self.canny_high, L2gradient=False)

        # Build float32 HWC4 in [0,1]
        # Note: ensure contiguous to avoid implicit copies later
        rgb_f = rgb.astype(np.float32) * (1.0 / 255.0)
        edge_f = edge.astype(np.float32) * (1.0 / 255.0)

        hwc4 = np.empty((h, w, 4), dtype=np.float32)
        hwc4[:, :, 0:3] = rgb_f
        hwc4[:, :, 3] = edge_f

        # Normalize: (x - mean) / std  == x*inv_std + bias
        hwc4 *= self.inv_std_np
        hwc4 += self.bias_np

        # To torch CHW (float32 CPU)
        t = torch.from_numpy(hwc4).permute(2, 0, 1).contiguous()
        return t



def load_video_as_tensor(
    video_path: str,
    max_frames: int | None = None,
) -> torch.Tensor:
    """
    Read video, convert each frame to 4ch tensor, stack into (N,4,224,224) on CPU.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    preproc_kind = (os.environ.get('ROAD_VISION_PREPROC', 'fast')).strip().lower()
    if preproc_kind in ('fast', 'cv2'):
        preproc = FourChannelPreprocessorFast(out_hw=(224, 224), canny_low=50, canny_high=150)
    else:
        preproc = FourChannelPreprocessor(out_hw=(224, 224), edge_low=50, edge_high=150)
    frames = []

    frame_count = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img4 = preproc(frame)
        frames.append(img4)
        frame_count += 1

        if (max_frames is not None) and (frame_count >= max_frames):
            break

    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames read from video.")

    x_cpu = torch.stack(frames, dim=0)  # (N,4,224,224)
    t1 = time.perf_counter()
    if TIMING:
        print(f"[timing] load+preproc: {t1 - t0:.3f}s for {frame_count} frames", flush=True)

    return x_cpu


# ----------------------------------------------------------------------
# Live inference (stream) with overlay
# ----------------------------------------------------------------------
def infer_video_live(
    video_path: str,
    device: torch.device,
    lite_ui: bool = False,
) -> None:
    """Stream video, run inference per frame, and display live with timeline UI."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    model = build_model(device)
    model.eval()
    preproc_kind = (os.environ.get('ROAD_VISION_PREPROC', 'fast')).strip().lower()
    if preproc_kind in ('fast', 'cv2'):
        preproc = FourChannelPreprocessorFast(out_hw=(224, 224), canny_low=50, canny_high=150)
    else:
        preproc = FourChannelPreprocessor(out_hw=(224, 224), edge_low=50, edge_high=150)

    use_pinned = (device.type != 'cpu') and (os.environ.get('ROAD_VISION_PINNED', '1') == '1')
    pin_buf = None
    dev_buf = None
    if use_pinned:
        # Reuse pinned host buffer + device buffer to avoid per-frame allocations.
        pin_buf = torch.empty((1, 4, 224, 224), dtype=torch.float32, pin_memory=True)
        dev_buf = torch.empty((1, 4, 224, 224), dtype=torch.float32, device=device)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1920 if not lite_ui else 1280, 1080 if not lite_ui else 720)
    font = FONT

    # 트랙바 콜백
    start_frame, end_frame, current, paused = 0, max(0, total-1), 0, False
    loop_fps = fps if fps > 0 else 0.0

    if not lite_ui:
        def on_timeline(pos): cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        def on_start(pos):
            nonlocal start_frame
            start_frame = min(pos, end_frame-1)
            cv2.setTrackbarPos("Start", WINDOW_NAME, start_frame)
        def on_end(pos):
            nonlocal end_frame
            end_frame = max(pos, start_frame+1)
            cv2.setTrackbarPos("End", WINDOW_NAME, end_frame)

        cv2.createTrackbar("Timeline", WINDOW_NAME, 0, max(total-1, 0), on_timeline)
        cv2.createTrackbar("Start",    WINDOW_NAME, 0, max(total-1, 0), on_start)
        cv2.createTrackbar("End",      WINDOW_NAME, max(total-1, 0), max(total-1, 0), on_end)

    infer_stage = "xpu" if device.type != "cpu" else "cpu"
    print(f"[device] using {infer_stage}", flush=True)
    print(f"[stage] read=cpu preproc=cpu infer={infer_stage} ui=cpu", flush=True)
    print("❚❚ Space:재생/일시정지 | A/D:±1프레임 | S/E:구간점 이동 | R:재추론 | Q:종료")

    last_ts = time.perf_counter()
    acc = {"read": 0.0, "preproc": 0.0, "infer": 0.0, "ui": 0.0, "frames": 0}

    while True:
        if paused:
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                paused = False
            elif not lite_ui and key in (ord('a'), 81):  # left arrow
                on_timeline(max(current - 1, 0))
            elif not lite_ui and key in (ord('d'), 83):  # right arrow
                on_timeline(min(current + 1, total - 1))
            elif not lite_ui and key == ord('s'):
                on_timeline(start_frame)
            elif not lite_ui and key == ord('e'):
                on_timeline(end_frame)
            elif not lite_ui and key == ord('r'):
                on_timeline(start_frame)
                paused = False
                print(f"▶ 구간 재추론: {start_frame} ~ {end_frame}")
            elif key == ord("q"):
                break
            continue

        now = time.perf_counter()
        dt = now - last_ts
        last_ts = now
        if dt > 0:
            loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / dt)

        t_r0 = time.perf_counter()
        ret, frame = cap.read()
        t_r1 = time.perf_counter()
        if not ret or current > end_frame:
            break

        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not lite_ui:
            cv2.setTrackbarPos("Timeline", WINDOW_NAME, current)

        t_p0 = time.perf_counter()
        img4_cpu = preproc(frame)  # [4,224,224] float32 CPU
        if use_pinned:
            pin_buf[0].copy_(img4_cpu, non_blocking=False)
            dev_buf.copy_(pin_buf, non_blocking=True)
            img4 = dev_buf
        else:
            img4 = img4_cpu.unsqueeze(0).to(device, non_blocking=(device.type != 'cpu'))
        t_p1 = time.perf_counter()

        t_f0 = time.perf_counter()
        with torch.no_grad():
            logits = model(img4)
            probs = F.softmax(logits, dim=1)[0]
            idx_pred = probs.argmax().item()
            conf = probs[idx_pred].item() * 100.0
        if TIMING and device.type == "xpu":
            torch.xpu.synchronize()
        t_f1 = time.perf_counter()

        t_u0 = time.perf_counter()
        label = label_map[idx_pred]
        h, w = frame.shape[:2]
        disp_scale = min(1.0, MAX_SCREEN_H / h)
        display_frame = cv2.resize(frame, (int(w * disp_scale), int(h * disp_scale)))
        dh = display_frame.shape[0]
        font_scale, thickness, y_pred, y_time = get_text_params(dh)

        txt = f"{label}: {conf:.1f}%"
        cv2.putText(display_frame, txt, (int(dh * 0.03), y_pred),
                    font, font_scale, BAR_COLOR, thickness, cv2.LINE_AA)

        def frames_to_time(fr, f):
            secs = int(fr / f) if f > 0 else 0
            return time.strftime('%H:%M:%S', time.gmtime(secs))

        if not lite_ui:
            info = f"{frames_to_time(current,fps)}/{frames_to_time(total,fps)}  " \
                   f"[{frames_to_time(start_frame,fps)} - {frames_to_time(end_frame,fps)}]"
            fps_txt = f"Video FPS: {fps:.1f} | Proc FPS: {loop_fps:.1f}"
            cv2.putText(display_frame, info, (int(dh * 0.03), y_time),
                        font, font_scale * 0.7, (255, 255, 255), thickness, cv2.LINE_AA)
            cv2.putText(display_frame, fps_txt, (int(dh * 0.03), int(y_time + dh * 0.04)),
                        font, font_scale * 0.7, (200, 200, 200), thickness, cv2.LINE_AA)
        else:
            fps_txt = f"Proc FPS: {loop_fps:.1f}"
            cv2.putText(display_frame, fps_txt, (int(dh * 0.03), int(y_time + dh * 0.04)),
                        font, font_scale * 0.7, (200, 200, 200), thickness, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        t_u1 = time.perf_counter()

        if TIMING:
            acc["read"] += (t_r1 - t_r0)
            acc["preproc"] += (t_p1 - t_p0)
            acc["infer"] += (t_f1 - t_f0)
            acc["ui"] += (t_u1 - t_u0)
            acc["frames"] += 1
            if acc["frames"] % TIMING_INT == 0:
                f = acc["frames"]
                read_ms = acc["read"] * 1000 / f
                pre_ms = acc["preproc"] * 1000 / f
                inf_ms = acc["infer"] * 1000 / f
                ui_ms = acc["ui"] * 1000 / f
                total_ms = read_ms + pre_ms + inf_ms + ui_ms
                fps_est = 1000.0 / total_ms if total_ms > 0 else 0.0
                print(
                    f"[timing] read={read_ms:.2f}ms "
                    f"preproc={pre_ms:.2f}ms "
                    f"infer={inf_ms:.2f}({infer_stage})ms "
                    f"ui={ui_ms:.2f}ms "
                    f"total={total_ms:.2f} [dev read=cpu preproc=cpu ui=cpu]ms "
                    f"fps≈{fps_est:.1f}",
                    flush=True,
                )
                acc = {"read": 0.0, "preproc": 0.0, "infer": 0.0, "ui": 0.0, "frames": 0}

        if key == ord(" "):
            paused = True
        elif not lite_ui and key in (ord('a'), 81):  # left arrow
            on_timeline(max(current - 1, 0))
        elif not lite_ui and key in (ord('d'), 83):  # right arrow
            on_timeline(min(current + 1, total - 1))
        elif not lite_ui and key == ord('s'):
            on_timeline(start_frame)
        elif not lite_ui and key == ord('e'):
            on_timeline(end_frame)
        elif not lite_ui and key == ord('r'):
            on_timeline(start_frame)
            paused = False
            print(f"▶ 구간 재추론: {start_frame} ~ {end_frame}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# Batch inference core
# ----------------------------------------------------------------------
def infer_video_batch(
    video_path: str,
    device: torch.device,
    max_frames: int | None = None,
    chunk_size: int | None = None,
):
    """
    Offline batch inference:
      1) Load video -> (N,4,224,224) on CPU
      2) Move to device in chunks
      3) Run CNN+GRU+MLP on each chunk
      4) Return per-frame labels and confidences
    """
    print(f"[info] video: {video_path}")
    x_cpu = load_video_as_tensor(video_path, max_frames=max_frames)
    N = x_cpu.shape[0]
    print(f"[info] frames loaded: {N}")

    model = build_model(device)
    print(f"[info] model device: {device}")

    if chunk_size is None or chunk_size >= N:
        chunk_size = N

    labels: list[str] = []
    confidences: list[float] = []

    model.eval()
    t_total0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_cpu[start:end].to(device, non_blocking=True)

            t0 = time.perf_counter() if TIMING else None
            logits = model(x_chunk)
            t1 = time.perf_counter() if TIMING else None

            probs = F.softmax(logits, dim=1)
            idxs = probs.argmax(dim=1)
            confs = probs.max(dim=1).values

            for idx, conf in zip(idxs.tolist(), confs.tolist()):
                labels.append(label_map[idx])
                confidences.append(conf * 100.0)

            if TIMING:
                print(
                    f"[chunk] {start}-{end-1} "
                    f"forward={t1 - t0:.3f}s "
                    f"B={end-start}",
                    flush=True,
                )
            else:
                print(f"[chunk] {start}-{end-1} done", flush=True)

    t_total1 = time.perf_counter()
    if TIMING:
        per_frame = (t_total1 - t_total0) / N
        fps = 1.0 / per_frame if per_frame > 0 else 0.0
        print(
            f"[timing] total inference: {t_total1 - t_total0:.3f}s, "
            f"per-frame={per_frame*1000:.2f}ms, fps≈{fps:.1f}",
            flush=True,
        )

    return labels, confidences


# ----------------------------------------------------------------------
# Visualization: replay video with overlays (and optional mp4 save)
# ----------------------------------------------------------------------
def replay_with_overlays(
    video_path: str,
    labels: list[str],
    confidences: list[float],
    save_path: str | None = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup writer if save_path given
    writer = None
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps if fps > 0 else 30.0, (width, height))

    cv2.namedWindow("Road-Vision-Batch", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Road-Vision-Batch", 1280, 720)

    font = cv2.FONT_HERSHEY_SIMPLEX

    idx = 0
    print("▶ Batch replay: Space=Pause/Play, Q=Quit")
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or idx >= len(labels):
                break

            label = labels[idx]
            conf  = confidences[idx]

            text = f"{idx:05d} | {label}: {conf:.1f}%"
            cv2.putText(
                frame,
                text,
                (int(width * 0.03), int(height * 0.08)),
                font,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Road-Vision-Batch", frame)
            if writer is not None:
                writer.write(frame)

            idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[info] saved: {save_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    import argparse
    from collections import Counter

    default_video = ROOT / "data" / "normal_road.mp4"

    parser = argparse.ArgumentParser(description="Offline batch inference on XPU/CPU")
    parser.add_argument(
        "video",
        nargs="?",
        default=str(default_video),
        help="Input video file path (mp4, avi, ...).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the maximum number of frames to load (for quick testing).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of frames per chunk when running on device.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display replay window.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="If set, save replay video with overlays to this path (e.g. out.mp4).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live streaming inference (read→preprocess→infer→show) instead of offline batch.",
    )
    parser.add_argument(
        "--live-lite",
        action="store_true",
        help="Live mode with minimal UI (no trackbars, lighter overlay) for max FPS.",
    )

    args = parser.parse_args()

    device = select_device()
    if args.live:
        infer_video_live(args.video, device=device, lite_ui=args.live_lite)
        return

    labels, confidences = infer_video_batch(
        args.video,
        device=device,
        max_frames=args.max_frames,
        chunk_size=args.chunk_size,
    )

    counter = Counter(labels)
    print("\n[summary] label distribution:")
    for k, v in counter.items():
        print(f"  {k}: {v} frames")

    print("\n[sample] first 10 frames:")
    for i in range(min(10, len(labels))):
        print(f"  frame {i}: {labels[i]} ({confidences[i]:.1f}%)")

    if not args.no_show or args.save_video is not None:
        save_path = args.save_video
        replay_with_overlays(args.video, labels, confidences, save_path=save_path)


if __name__ == "__main__":
    main()
