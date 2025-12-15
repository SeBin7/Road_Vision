# src/inference/infer_batch_xpu.py
# Offline batch inference + live streaming on XPU/CPU
# - Batch mode: load full video, preprocess to 4ch, run CNN+GRU on XPU/CPU,
#               then replay video with overlayed predictions.
# - Live mode : read→preprocess→infer→UI per frame (like infer_4k) with timing & mem debug.

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
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

FORCE_CPU   = os.environ.get("ROAD_VISION_CPU", "").lower() in {"1", "true", "yes"}
DEBUG       = os.environ.get("ROAD_VISION_DEBUG", "").lower() in {"1", "true", "yes"}
TIMING      = DEBUG or os.environ.get("ROAD_VISION_TIMING", "").lower() in {"1", "true", "yes"}
TIMING_INT  = int(os.environ.get("ROAD_VISION_TIMING_INT", "30"))

# XPU memory debug
MEM_DEBUG   = os.environ.get("ROAD_VISION_MEM_DEBUG", "").lower() in {"1", "true", "yes"}
MEM_INT     = int(os.environ.get("ROAD_VISION_MEM_INT", "300"))

# UI/look constants (align with infer_4k)
SEQ_LEN      = 1
WINDOW_NAME  = "Road-Vision-Live"
BAR_COLOR    = (0, 255, 0)  # BGR
FONT         = cv2.FONT_HERSHEY_SIMPLEX
MAX_SCREEN_H = 1080


def _dbg(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)


def get_text_params(h: int):
    """Return (font_scale, thickness, y_pred, y_time) based on frame height."""
    font_scale = h / 1080 * 1.0          # 1080p→1.0, 4K(2160p)→2.0
    thickness  = max(1, int(h / 1080 * 2))
    y_pred     = int(h * 0.04)           # prediction text y
    y_time     = int(h * 0.08)           # time/FPS text y
    return font_scale, thickness, y_pred, y_time


# ----------------------------------------------------------------------
# Device & memory helpers
# ----------------------------------------------------------------------
def select_device() -> torch.device:
    """Select XPU if available and not forced to CPU."""
    if (not FORCE_CPU) and torch.xpu.is_available():
        dev = torch.device("xpu")
    else:
        dev = torch.device("cpu")
    if DEBUG or TIMING:
        print(f"[device] using {dev}", flush=True)
    return dev


def _print_xpu_memory(tag: str, device: torch.device) -> None:
    """Print XPU memory usage (allocated/reserved) if on XPU and mem debug enabled."""
    if not MEM_DEBUG:
        return
    if device.type != "xpu":
        return
    torch.xpu.synchronize()
    alloc = torch.xpu.memory_allocated() / (1024**2)
    reserv = torch.xpu.memory_reserved() / (1024**2)
    print(f"[mem] {tag}: alloc={alloc:.1f}MB reserved={reserv:.1f}MB", flush=True)


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
    # Fallback: use first name at project root
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

    if DEBUG:
        first_param_dev = next(model.parameters()).device
        print(f"[debug] model first param device: {first_param_dev}", flush=True)

    _print_xpu_memory("after model build", device)
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

    preproc = FourChannelPreprocessor()
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
    """
    Stream video, run inference per frame, and display live with timeline UI.

    Timing breakdown (ms/frame):
      - read:    video decoding (cap.read)
      - preproc: RGB/resize/edge/normalize + H2D copy
      - infer:   CNN+GRU+MLP+softmax
      - ui:      resize for display, overlay text, imshow, waitKey
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    model = build_model(device)
    model.eval()
    preproc = FourChannelPreprocessor()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1920 if not lite_ui else 1280, 1080 if not lite_ui else 720)
    font = FONT

    # Trackbar-related state (full UI only)
    start_frame, end_frame, current, paused = 0, max(0, total - 1), 0, False
    loop_fps = fps if fps > 0 else 0.0

    if not lite_ui:
        def on_timeline(pos):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        def on_start(pos):
            nonlocal start_frame
            start_frame = min(pos, end_frame - 1)
            cv2.setTrackbarPos("Start", WINDOW_NAME, start_frame)

        def on_end(pos):
            nonlocal end_frame
            end_frame = max(pos, start_frame + 1)
            cv2.setTrackbarPos("End", WINDOW_NAME, end_frame)

        cv2.createTrackbar("Timeline", WINDOW_NAME, 0, max(total - 1, 0), on_timeline)
        cv2.createTrackbar("Start",    WINDOW_NAME, 0, max(total - 1, 0), on_start)
        cv2.createTrackbar("End",      WINDOW_NAME, max(total - 1, 0), max(total - 1, 0), on_end)

    print("❚❚ Space:재생/일시정지 | A/D:±1프레임 | S/E:구간점 이동 | R:재추론 | Q:종료")
    print(f"[info] video='{video_path}', frames={total}, fps={fps:.2f}")

    # Loop timing (for Proc FPS in UI)
    loop_last_ts = time.perf_counter()

    # Aggregated timing for breakdown
    acc = {"read": 0.0, "preproc": 0.0, "infer": 0.0, "ui": 0.0, "frames": 0}
    # XPU memory debug counter
    frame_counter_for_mem = 0

    while True:
        # ---------------- Read + preprocess + infer ----------------
        if not paused:
            # Loop FPS (instant, used only for overlay)
            now = time.perf_counter()
            dt = now - loop_last_ts
            loop_last_ts = now
            if dt > 0:
                loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / dt)

            # read
            t_r0 = time.perf_counter()
            ret, frame = cap.read()
            t_r1 = time.perf_counter()
            if not ret or current > end_frame:
                print("[info] end of video or segment")
                break

            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if not lite_ui:
                cv2.setTrackbarPos("Timeline", WINDOW_NAME, current)

            # preproc
            t_p0 = time.perf_counter()
            img4 = preproc(frame)  # CPU (4,224,224)
            if device.type != "cpu":
                img4 = img4.pin_memory()
            img4 = img4.unsqueeze(0).to(device, non_blocking=True)  # (1,4,224,224)
            t_p1 = time.perf_counter()

            # Check device of input tensor once
            if DEBUG and acc["frames"] == 0:
                print(f"[debug] img4 device: {img4.device}", flush=True)

            # infer
            with torch.no_grad():
                t_f0 = time.perf_counter()
                logits = model(img4)
                probs = F.softmax(logits, dim=1)[0]
                idx_pred = probs.argmax().item()
                conf = probs[idx_pred].item() * 100.0
                t_f1 = time.perf_counter()

            label = label_map[idx_pred]

            # XPU memory snapshot (optional)
            frame_counter_for_mem += 1
            if MEM_DEBUG and (frame_counter_for_mem % MEM_INT == 0):
                _print_xpu_memory(f"live frame {frame_counter_for_mem}", device)

            # --------------------- UI (overlay + imshow) ---------------------
            h, w = frame.shape[:2]
            disp_scale = min(1.0, MAX_SCREEN_H / h)
            display_frame = cv2.resize(frame, (int(w * disp_scale), int(h * disp_scale)))
            dh = display_frame.shape[0]
            font_scale, thickness, y_pred, y_time = get_text_params(dh)

            txt = f"{label}: {conf:.1f}%"
            cv2.putText(
                display_frame,
                txt,
                (int(dh * 0.03), y_pred),
                font,
                font_scale,
                BAR_COLOR,
                thickness,
                cv2.LINE_AA,
            )

            def frames_to_time(fr, f):
                secs = int(fr / f) if f > 0 else 0
                return time.strftime("%H:%M:%S", time.gmtime(secs))

            if not lite_ui:
                info = (
                    f"{frames_to_time(current, fps)}/"
                    f"{frames_to_time(total, fps)}  "
                    f"[{frames_to_time(start_frame, fps)} - {frames_to_time(end_frame, fps)}]"
                )
                fps_txt = f"Video FPS: {fps:.1f} | Proc FPS: {loop_fps:.1f}"
                cv2.putText(
                    display_frame,
                    info,
                    (int(dh * 0.03), y_time),
                    font,
                    font_scale * 0.7,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    fps_txt,
                    (int(dh * 0.03), int(y_time + dh * 0.04)),
                    font,
                    font_scale * 0.7,
                    (200, 200, 200),
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                fps_txt = f"Proc FPS: {loop_fps:.1f}"
                cv2.putText(
                    display_frame,
                    fps_txt,
                    (int(dh * 0.03), int(y_time + dh * 0.04)),
                    font,
                    font_scale * 0.7,
                    (200, 200, 200),
                    thickness,
                    cv2.LINE_AA,
                )

            t_u0 = time.perf_counter()
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            t_u1 = time.perf_counter()

            # --------------------- Timing accumulation ---------------------
            if TIMING:
                acc["read"] += (t_r1 - t_r0)
                acc["preproc"] += (t_p1 - t_p0)
                acc["infer"] += (t_f1 - t_f0)
                acc["ui"] += (t_u1 - t_u0)
                acc["frames"] += 1

                if acc["frames"] % TIMING_INT == 0:
                    f = acc["frames"]
                    avg_read = acc["read"] * 1000.0 / f
                    avg_pre  = acc["preproc"] * 1000.0 / f
                    avg_inf  = acc["infer"] * 1000.0 / f
                    avg_ui   = acc["ui"] * 1000.0 / f

                    avg_total = (acc["read"] + acc["preproc"] + acc["infer"] + acc["ui"]) / f
                    real_fps = 1.0 / avg_total if avg_total > 0 else 0.0

                    print(
                        f"[timing] read={avg_read:.2f}ms "
                        f"preproc={avg_pre:.2f}ms "
                        f"infer={avg_inf:.2f}ms "
                        f"ui={avg_ui:.2f}ms "
                        f"total={avg_total*1000.0:.2f}ms "
                        f"fps≈{real_fps:.1f}",
                        flush=True,
                    )

        else:
            # paused: still need key handling
            key = cv2.waitKey(1) & 0xFF

        # --------------------- Key handling ---------------------
        if key == ord(" "):
            paused = not paused

        elif not lite_ui and key in (ord("a"), 81):  # 'a' or left arrow
            new_pos = max(current - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            current = new_pos

        elif not lite_ui and key in (ord("d"), 83):  # 'd' or right arrow
            new_pos = min(current + 1, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            current = new_pos

        elif not lite_ui and key == ord("s"):        # jump to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current = start_frame
            paused = False

        elif not lite_ui and key == ord("e"):        # jump to end
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
            current = end_frame
            paused = False

        elif not lite_ui and key == ord("r"):        # replay segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current = start_frame
            paused = False
            print(f"▶ segment replay: {start_frame} ~ {end_frame}")

        elif key == ord("q"):
            print("[info] quit by user")
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

    # Default chunk size: N (all frames) if not specified
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
            if DEBUG and start == 0:
                print(f"[debug] x_chunk device: {x_chunk.device}", flush=True)

            _print_xpu_memory(f"batch frames {start}-{end-1} before forward", device)

            t0 = time.perf_counter() if TIMING else None
            logits = model(x_chunk)
            t1 = time.perf_counter() if TIMING else None

            _print_xpu_memory(f"batch frames {start}-{end-1} after forward", device)

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

    parser = argparse.ArgumentParser(description="Offline batch + live inference on XPU/CPU")
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
        help="Number of frames per chunk when running on device (batch mode).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display replay window (batch mode).",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="If set, save replay video with overlays to this path (e.g. out.mp4) in batch mode.",
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
