"""
OpenVINO(INT8) CNN + Torch GRU/MLP inference pipeline (CPU).

- CNN: models/ir_int8/model.xml (INT8, 4채널 입력)
- Classifier: best_gru_mlp_classifier.pth (Torch, CPU)
- 입력: 영상 파일 (기본 data/normal_road.mp4)
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from openvino.runtime import Core
from torchvision import transforms

# sys.path에 src 추가
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU  # noqa: E402

# ─────────────── 설정 ───────────────
SEQ_LEN = 5
WINDOW_NAME = "Road-Vision (IR+Torch)"
BAR_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_SCREEN_H = 1080

label_map = {0: "broken", 1: "normal_road", 2: "snow_road", 3: "wet_road"}
device = torch.device("cpu")

ROOT = Path(__file__).resolve().parents[2]
IR_PATH = ROOT / "models" / "ir_int8" / "model.xml"
DEFAULT_GRU_WEIGHT = ROOT / "models" / "models" / "gru_mlp_classifier_4ch_val.pth"
GRU_WEIGHT_NAMES = [
    "gru_mlp_classifier_4ch_val.pth",
    "best_gru_mlp_classifier_4ch.pth",
    "best_gru_mlp_classifier.pth",
]
MODEL_DIRS = [
    ROOT / "models" / "models",
    ROOT / "models" / "server_model",
    ROOT / "models" / "4ch_results",
    ROOT,
]


def find_weight(names: List[str]) -> Path:
    for name in names:
        for base in MODEL_DIRS:
            cand = base / name
            if cand.exists():
                print(f"[+] GRU weight: {cand}")
                return cand
    raise FileNotFoundError(f"Classifier weight not found in: {names}")


def frames_to_time(frames, fps):
    secs = int(frames / fps)
    return time.strftime("%H:%M:%S", time.gmtime(secs))


class VideoInference:
    def __init__(self, seq_len: int = SEQ_LEN, gru_path: Path | None = None):
        self.seq_len = seq_len
        self.buffer: Deque[torch.Tensor] = deque(maxlen=seq_len)  # feature buffer

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # OpenVINO CNN (INT8)
        core = Core()
        if not IR_PATH.exists():
            raise FileNotFoundError(IR_PATH)
        self.compiled = core.compile_model(str(IR_PATH), "CPU")
        self.input_name = self.compiled.input(0).get_any_name()
        self.output_name = self.compiled.output(0).get_any_name()

        # Torch classifier
        cls_path = Path(gru_path) if gru_path else find_weight(GRU_WEIGHT_NAMES)
        self.cls = GRU_MLP_Classifier_XPU(feature_dim=128).to(device)
        self._load_classifier(cls_path)
        self.cls.eval()

    def _load_classifier(self, path: Path) -> None:
        """Load classifier weights, remapping old key names if needed."""
        state = torch.load(path, map_location=device)
        try:
            self.cls.load_state_dict(state)
            return
        except RuntimeError:
            # handle older checkpoints without dropout (linear at index 2)
            remapped = {}
            for k, v in state.items():
                if k.startswith("classifier.2.") and ("weight" in k or "bias" in k):
                    new_k = k.replace("classifier.2.", "classifier.3.")
                    remapped[new_k] = v
                elif k.startswith("classifier.2."):
                    continue  # skip other classifier.2 entries
                remapped[k] = v
            self.cls.load_state_dict(remapped, strict=False)

    def _push_frame(self, frame_bgr):
        # RGB → Tensor
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t_rgb = self.transform(pil)  # (3,224,224)
        # Canny edge 채널 생성
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.resize(edges, (224, 224))
        t_edge = torch.from_numpy(edges).unsqueeze(0).float().div(255.0)  # (1,224,224)
        # 4채널 결합 후 Normalize
        img4 = torch.cat([t_rgb, t_edge], dim=0)  # (4,224,224)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5], device=img4.device)
        std = torch.tensor([0.229, 0.224, 0.225, 0.5], device=img4.device)
        img4 = (img4 - mean[:, None, None]) / std[:, None, None]

        # OpenVINO infer (expect NHWC? we feed NCHW float32)
        np_in = img4.unsqueeze(0).cpu().numpy()  # (1,4,224,224)
        feat_np = self.compiled({self.input_name: np_in})[self.output_name]
        feat = torch.from_numpy(feat_np).squeeze(0)  # (128,)
        self.buffer.append(feat)

    def predict(self):
        if len(self.buffer) < self.seq_len:
            return None, None
        x = torch.stack(list(self.buffer)).unsqueeze(0)  # (1,T,128)
        with torch.no_grad():
            logit = self.cls(x.to(device))
            prob = F.softmax(logit, dim=1)[0]
            idx = torch.argmax(prob).item()
            return label_map[idx], prob[idx].item() * 100


def main(video_path: str, gru_path: Path | None = None):
    video_path = str(video_path)
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    loop_fps = fps
    last_ts = time.perf_counter()
    start_frame, end_frame, current, paused = 0, total - 1, 0, False

    infer = VideoInference(gru_path=gru_path)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1920, 1080)

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

    cv2.createTrackbar("Timeline", WINDOW_NAME, 0, total - 1, on_timeline)
    cv2.createTrackbar("Start", WINDOW_NAME, 0, total - 1, on_start)
    cv2.createTrackbar("End", WINDOW_NAME, total - 1, total - 1, on_end)

    print("❚❚ Space:재생/일시정지 | A/D:±1프레임 | S/E:구간점 이동 | R:재추론 | Q:종료")

    while True:
        if not paused:
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / dt)

            ret, frame = cap.read()
            if not ret or current > end_frame:
                break
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            cv2.setTrackbarPos("Timeline", WINDOW_NAME, current)

            if start_frame <= current <= end_frame:
                infer._push_frame(frame)
                label, conf = infer.predict()
            else:
                label, conf = None, None

            h, w = frame.shape[:2]
            disp_scale = min(1.0, MAX_SCREEN_H / h)
            display_frame = cv2.resize(frame, (int(w * disp_scale), int(h * disp_scale)))

            if label:
                txt = f"{label}: {conf:.1f}%"
                cv2.putText(display_frame, txt, (10, 40), FONT, 1.0, BAR_COLOR, 2, cv2.LINE_AA)

            info = f"{frames_to_time(current,fps)}/{frames_to_time(total,fps)}  " \
                   f"[{frames_to_time(start_frame,fps)} - {frames_to_time(end_frame,fps)}]"
            fps_txt = f"Video FPS: {fps:.1f} | Proc FPS: {loop_fps:.1f}"
            cv2.putText(display_frame, info, (10, 75), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, fps_txt, (10, 105), FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            paused = not paused
        elif key in (ord("a"), 81):
            on_timeline(max(current - 1, 0))
        elif key in (ord("d"), 83):
            on_timeline(min(current + 1, total - 1))
        elif key == ord("s"):
            on_timeline(start_frame)
        elif key == ord("e"):
            on_timeline(end_frame)
        elif key == ord("r"):
            infer.buffer.clear()
            on_timeline(start_frame)
            paused = False
            print(f"▶ 구간 재추론: {frames_to_time(start_frame,fps)} ~ {frames_to_time(end_frame,fps)}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    default_video = ROOT / "data" / "normal_road.mp4"
    parser = argparse.ArgumentParser(description="Road-Vision IR(Torch GRU) timeline inference")
    parser.add_argument("video", nargs="?", default=str(default_video), help="입력 동영상 파일")
    parser.add_argument("--gru-weight", type=str, default=str(DEFAULT_GRU_WEIGHT),
                        help="GRU/MLP 가중치 경로 지정")
    args = parser.parse_args()
    gru_path = Path(args.gru_weight) if args.gru_weight else None
    main(args.video, gru_path=gru_path)
