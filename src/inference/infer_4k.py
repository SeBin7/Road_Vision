# ───────────────────────────────────────────────────────────────────────────────
# video_timeline_infer.py  ── 4K-friendly display 버전
# ───────────────────────────────────────────────────────────────────────────────
import os

# OpenCV 패키지에 Wayland 플러그인이 없으므로 xcb로 강제해 UI 크래시 방지
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2, glob, time, torch, torch.nn.functional as F
from PIL import Image
from collections import deque
from pathlib import Path
import sys
from torchvision import transforms

# Ensure project src/ is on sys.path so `models` can be imported when run as a script
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.Mobilenet_hailo_4ch import MobileNetFeatureExtractor
from models.CNN import CNNFeatureExtractor
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

import torch

# ─────────────── 하이퍼파라미터 ───────────────
SEQ_LEN      = 10
WINDOW_NAME  = 'Road-Vision'
BAR_COLOR    = (  0,255,  0)  # BGR
FONT         = cv2.FONT_HERSHEY_SIMPLEX
MAX_SCREEN_H = 1080           # 모니터 세로 해상도(필요시 수정)

label_map = {0: 'broken', 1: 'normal_road', 2: 'snow_road', 3: 'wet_road'}
#label_map = {0: 'normal_road', 1: 'snow_road', 2: 'wet_road'}  # 예시로 수정
FORCE_CPU = os.environ.get("ROAD_VISION_CPU", "").lower() in {"1", "true", "yes"}
# 학습 시 구조: CNN=XPU, GRU/MLP=CPU. 별도 지정 없으면 이를 기본값으로 사용.
cls_cpu_env = os.environ.get("ROAD_VISION_CLS_CPU")
if cls_cpu_env is None:
    CLS_CPU = True
else:
    CLS_CPU = cls_cpu_env.lower() not in {"0", "false", "no"}

SOFTMAX_CPU_ENV = os.environ.get("ROAD_VISION_SOFTMAX_CPU", "").lower() in {"1", "true", "yes"}

DEBUG_LOG  = os.environ.get("ROAD_VISION_DEBUG", "").lower() in {"1", "true", "yes"}
TIMING     = DEBUG_LOG or os.environ.get("ROAD_VISION_TIMING", "").lower() in {"1", "true", "yes"}
TIMING_INT = int(os.environ.get("ROAD_VISION_TIMING_INT", "30"))

if FORCE_CPU or not torch.xpu.is_available():
    CNN_DEVICE = torch.device("cpu")
    CLS_DEVICE = torch.device("cpu")
elif CLS_CPU:
    # 기본: CNN만 XPU, GRU/MLP는 CPU
    CNN_DEVICE = torch.device("xpu")
    CLS_DEVICE = torch.device("cpu")
else:
    # 옵션: 둘 다 XPU
    CNN_DEVICE = torch.device("xpu")
    CLS_DEVICE = torch.device("xpu")

# softmax 위치: 기본은 CPU(안정성/속도). 환경변수로 XPU 강제 가능.
SOFTMAX_DEVICE = CLS_DEVICE if SOFTMAX_CPU_ENV is False else torch.device("cpu")

def _dbg(msg):
    if DEBUG_LOG:
        print(msg, flush=True)
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


def _find_model_file(names) -> Path:
    """Return the first existing path among expected model dirs and names."""
    for name in names:
        for base in MODEL_DIR_CANDIDATES:
            candidate = base / name
            if candidate.exists():
                return candidate
    return ROOT / names[0]  # fallback to first requested name

# ─────────────── 동적 텍스트 파라미터 ───────────────
def get_text_params(h):
    """프레임 높이에 비례한 (폰트스케일, 두께, 위치 y오프셋) 반환"""
    font_scale = h / 1080 * 1.0          # 1080p→1.0, 4K(2160p)→2.0
    thickness  = max(1, int(h / 1080 * 2))
    y_pred     = int(h * 0.04)           # 상태바 y위치
    y_time     = int(h * 0.08)           # 시간바 y위치
    return font_scale, thickness, y_pred, y_time

# ─────────────── 추론 클래스 ───────────────
class VideoInference:
    def __init__(self, seq_len=SEQ_LEN, quantize_cls: bool = False):
        self.seq_len = seq_len
        self.buffer  = deque(maxlen=seq_len)  # feature buffer (T, feature_dim)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        if DEBUG_LOG or TIMING:
            print(f"[device] CNN={CNN_DEVICE}, CLS={CLS_DEVICE}, softmax={SOFTMAX_DEVICE}", flush=True)
        self.cnn = MobileNetFeatureExtractor().to(CNN_DEVICE)
        self.cls = GRU(feature_dim=128).to(CLS_DEVICE)

        cnn_path = _find_model_file(CNN_WEIGHT_NAMES)
        cls_path = _find_model_file(CLS_WEIGHT_NAMES)

        self.cnn.load_state_dict(torch.load(cnn_path, map_location=CNN_DEVICE))
        self.cls.load_state_dict(torch.load(cls_path, map_location=CLS_DEVICE))
        self.cnn.eval()
        self.cls.eval()

        # 타이밍 수집용 스탯
        self.timing = TIMING
        self.timing_stats = {
            "read": 0.0,
            "preproc": 0.0,
            "cnn": 0.0,
            "cls": 0.0,
            "softmax": 0.0,
            "frames": 0,
        }

        # CPU에서 MLP/GRU만 동적 양자화(가벼운 속도 업)
        if quantize_cls and CLS_DEVICE.type == "cpu":
            import torch.ao.quantization as tq
            torch.backends.quantized.engine = "fbgemm"
            self.cls = tq.quantize_dynamic(self.cls, {torch.nn.Linear}, dtype=torch.qint8)
        # CPU일 때는 TorchScript로 고정해 Python 오버헤드 줄임
        if CLS_DEVICE.type == "cpu":
            self.cls = torch.jit.script(self.cls).eval()

        # XPU는 첫 호출 때 커널 JIT/컴파일로 몇 초 멈출 수 있으니 사전 웜업
        with torch.no_grad():
            if CNN_DEVICE.type == "xpu":
                dummy = torch.zeros(1, 4, 224, 224, device=CNN_DEVICE)
                _dbg("warmup: cnn start")
                _ = self.cnn(dummy)
                _dbg("warmup: cnn done")
            if CLS_DEVICE.type == "xpu":
                dummy_feat = torch.zeros(1, seq_len, 128, device=CLS_DEVICE)
                _dbg("warmup: cls start")
                _ = self.cls(dummy_feat)
                _dbg("warmup: cls done")

    def _report_timing(self):
        """주기적으로 단계별 평균 ms를 표시"""
        if not self.timing: return
        f = max(1, self.timing_stats["frames"])
        msg = (
            f"[timing] read={self.timing_stats['read']*1000/f:.2f}ms "
            f"preproc={self.timing_stats['preproc']*1000/f:.2f}ms "
            f"cnn={self.timing_stats['cnn']*1000/f:.2f}ms "
            f"cls={self.timing_stats['cls']*1000/f:.2f}ms "
            f"softmax={self.timing_stats['softmax']*1000/f:.2f}ms "
            f"frames={self.timing_stats['frames']}"
        )
        print(msg, flush=True)

    def _push_frame(self, frame_bgr):
        _dbg("push: bgr->rgb")
        t0 = time.perf_counter() if self.timing else None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        t_rgb = self.transform(pil)  # (3,224,224)

        _dbg("push: canny")
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.resize(edges, (224,224))
        t_edge = torch.from_numpy(edges).unsqueeze(0).float().div(255.0)  # (1,224,224)

        _dbg("push: cat/normalize")
        img4 = torch.cat([t_rgb, t_edge], dim=0)  # (4,224,224)
        mean = torch.tensor([0.485,0.456,0.406,0.5], device=img4.device)
        std  = torch.tensor([0.229,0.224,0.225,0.5], device=img4.device)
        img4 = (img4 - mean[:,None,None]) / std[:,None,None]
        if CNN_DEVICE.type != "cpu":
            img4 = img4.pin_memory()  # 호스트→디바이스 전송 오버랩 대비

        _dbg("push: to device")
        img4 = img4.unsqueeze(0).to(CNN_DEVICE, non_blocking=True)
        if self.timing:
            t1 = time.perf_counter()
            self.timing_stats["preproc"] += (t1 - t0)

        _dbg("push: cnn forward")
        with torch.no_grad():
            feat = self.cnn(img4).squeeze(0).detach()
        if self.timing:
            t2 = time.perf_counter()
            self.timing_stats["cnn"] += (t2 - t1)

        _dbg("push: move feat to cls device & append")
        self.buffer.append(feat.to(CLS_DEVICE, non_blocking=True))

    def predict(self):
        if len(self.buffer) < self.seq_len:
            return None, None

        _dbg("predict: stack buffer")
        x = torch.stack(list(self.buffer)).unsqueeze(0).to(CLS_DEVICE)   # (1,T,feature_dim)

        _dbg("predict: cls forward")
        with torch.no_grad():
            t0 = time.perf_counter() if self.timing else None
            logit = self.cls(x)
            t1 = time.perf_counter() if self.timing else None

        _dbg(f"predict: softmax on {SOFTMAX_DEVICE}")
        logit_sm = logit.to(SOFTMAX_DEVICE, non_blocking=True)
        t2 = time.perf_counter() if self.timing else None
        prob  = F.softmax(logit_sm, dim=1)[0]
        idx   = torch.argmax(prob).item()
        t3 = time.perf_counter() if self.timing else None

        if self.timing:
            self.timing_stats["cls"] += (t1 - t0)
            self.timing_stats["softmax"] += (t3 - t2)

        _dbg("predict: done")
        return label_map[idx], prob[idx].item()*100

# ─────────────── 유틸리티 ───────────────
def frames_to_time(frames, fps):
    secs = int(frames / fps)
    return time.strftime('%H:%M:%S', time.gmtime(secs))

# ─────────────── 메인 루틴 ───────────────
def main(video_path: str, quantize_cls: bool = False):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    loop_fps = fps  # 표시용 추정 처리 FPS(초기값은 메타데이터)
    last_ts = time.perf_counter()
    start_frame, end_frame, current, paused = 0, total-1, 0, False
    infer = VideoInference(quantize_cls=quantize_cls)

    # ❶ 창 크기 자유 + 비율 유지
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    cv2.resizeWindow(WINDOW_NAME, 1920, 1080) # 모니터 화면 크기 지정 1080p

    # cv2.setWindowProperty(WINDOW_NAME,
    #                   cv2.WND_PROP_FULLSCREEN,
    #                   cv2.WINDOW_FULLSCREEN) # 전체 화면 모드

    # ❷ 트랙바 콜백 정의
    def on_timeline(pos): cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    def on_start(pos):
        nonlocal start_frame; start_frame = min(pos, end_frame-1)
        cv2.setTrackbarPos('Start', WINDOW_NAME, start_frame)
    def on_end(pos):
        nonlocal end_frame; end_frame = max(pos, start_frame+1)
        cv2.setTrackbarPos('End', WINDOW_NAME, end_frame)

    cv2.createTrackbar('Timeline', WINDOW_NAME, 0, total-1, on_timeline)
    cv2.createTrackbar('Start',    WINDOW_NAME, 0, total-1, on_start)
    cv2.createTrackbar('End',      WINDOW_NAME, total-1, total-1, on_end)

    print('❚❚ Space:재생/일시정지 | A/D:±1프레임 | S/E:구간점 이동 | R:재추론 | Q:종료')

    while True:
        if not paused:
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                loop_fps = 0.9 * loop_fps + 0.1 * (1.0 / dt)  # 느리게 가중 평균

            t_read0 = time.perf_counter() if infer.timing else None
            ret, frame = cap.read()
            t_read1 = time.perf_counter() if infer.timing else None
            if not ret or current > end_frame: break
            current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            cv2.setTrackbarPos('Timeline', WINDOW_NAME, current)
            _dbg(f"read frame {current}")
            if infer.timing:
                infer.timing_stats["read"] += (t_read1 - t_read0)

            # ❸ 추론 버퍼 갱신
            if start_frame <= current <= end_frame:
                _dbg(f"push frame {current}")
                infer._push_frame(frame)
                label, conf = infer.predict()
                _dbg(f"infer frame {current} -> {label}, {conf}")

            # ❹ ------- DISPLAY 영역 (해상도 대응) -----------------
            h, w = frame.shape[:2]
            # 최대 화면 세로 `MAX_SCREEN_H`에 맞춰 축소 (예: 모니터 1080p → 0.5 스케일)
            disp_scale = min(1.0, MAX_SCREEN_H / h)
            display_frame = cv2.resize(frame, (int(w*disp_scale), int(h*disp_scale)))

            dh = display_frame.shape[0]  # 축소 후 높이
            font_scale, thickness, y_pred, y_time = get_text_params(dh)

            if label:
                txt = f'{label}: {conf:.1f}%'
                cv2.putText(display_frame, txt, (int(dh*0.03), y_pred),
                            FONT, font_scale, BAR_COLOR, thickness, cv2.LINE_AA)

            info = f'{frames_to_time(current,fps)}/{frames_to_time(total,fps)}  ' \
                   f'[{frames_to_time(start_frame,fps)} - {frames_to_time(end_frame,fps)}]'
            fps_txt = f'Video FPS: {fps:.1f} | Proc FPS: {loop_fps:.1f}'
            cv2.putText(display_frame, info, (int(dh*0.03), y_time),
                        FONT, font_scale*0.7, (255,255,255), thickness, cv2.LINE_AA)
            cv2.putText(display_frame, fps_txt, (int(dh*0.03), int(y_time + dh*0.04)),
                        FONT, font_scale*0.7, (200,200,200), thickness, cv2.LINE_AA)
            # -------------------------------------------------------

            cv2.imshow(WINDOW_NAME, display_frame)
            _dbg(f"imshow frame {current}")

            if infer.timing:
                infer.timing_stats["frames"] += 1
                if infer.timing_stats["frames"] % TIMING_INT == 0:
                    infer._report_timing()

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): paused = not paused
        elif key in (ord('a'), 81):  on_timeline(max(current-1, 0))
        elif key in (ord('d'), 83):  on_timeline(min(current+1, total-1))
        elif key == ord('s'):        on_timeline(start_frame)
        elif key == ord('e'):        on_timeline(end_frame)
        elif key == ord('r'):
            infer.buffer.clear(); on_timeline(start_frame); paused=False
            print(f'▶ 구간 재추론: {frames_to_time(start_frame,fps)} ~ {frames_to_time(end_frame,fps)}')
        elif key == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

# ─────────────── CLI ───────────────
if __name__ == '__main__':
    import argparse, sys
    default_video = ROOT / "data" / "normal_road.mp4"
    parser = argparse.ArgumentParser(description='Road-Vision timeline inference')
    parser.add_argument('video', nargs='?', default=str(default_video),
                        help='입력 동영상 파일(.mp4, .avi …). 생략 시 data/normal_road.mp4 사용')
    parser.add_argument('--quantize-cls', action='store_true',
                        help='CPU에서 GRU/MLP 분류기만 동적 양자화해 약간의 속도 향상')
    args = parser.parse_args()
    main(args.video, quantize_cls=args.quantize_cls)
