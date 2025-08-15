import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from collections import deque
from torchvision import transforms
from PIL import Image
from models.Mobilenet_5ch import MobileNetFeatureExtractor
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU
#from Dataset_5ch import compute_reflection_map, compute_edge_map

# ────────────────────────────── 설정 ──────────────────────────────
SEQ_LEN = 10
FEATURE_DIM = 128
NUM_CLASSES = 4
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
LABEL_MAP = {0: "broken", 1: "normal", 2: "snow", 3: "wet"}
MODEL_CNN_PATH = "best_cnn_feature_extractor.pth"
MODEL_GRU_PATH = "best_gru_mlp_classifier.pth"

def create_seekbar_overlay(frame, current_frame, total_frames, bar_height=20):
    h, w, _ = frame.shape
    seekbar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    ratio = current_frame / total_frames if total_frames > 0 else 0
    progress_width = int(ratio * w)
    seekbar[:, :progress_width] = (0, 255, 0)
    seekbar[:, progress_width:] = (50, 50, 50)
    combined = np.vstack((frame, seekbar))
    return combined

def on_trackbar(val):
    global cur_frame, cap
    cur_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

def compute_reflection_map(img_rgb, mode="LAB"):
    if mode == "RGB":
        gray = np.mean(img_rgb / 255.0, axis=2, keepdims=True)
        refl = (gray > 0.85).astype(np.float32)
    elif mode == "LAB":
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0:1] / 255.0
        refl = (L > 0.85).astype(np.float32)
    elif mode == "HSV":
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        V = hsv[:, :, 2:3] / 255.0
        refl = (V > 0.85).astype(np.float32)
    else:
        raise ValueError("Invalid refl_mode")
    return torch.from_numpy(refl).permute(2, 0, 1)  # (1, H, W)

def compute_edge_map(img_rgb, mode="canny"):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if mode == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag = np.clip(grad_mag / grad_mag.max(), 0, 1).astype(np.float32)
    elif mode == "canny":
        edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        grad_mag = edge
    else:
        raise ValueError("Invalid edge_mode")
    return torch.from_numpy(grad_mag).unsqueeze(0)  # (1, H, W)

# ────────────────────────────── 추론 클래스 ──────────────────────────────
class RoadInferencer:
    def __init__(self, transform):
        self.transform = transform
        self.window = deque(maxlen=SEQ_LEN)

        self.cnn = MobileNetFeatureExtractor(in_channels=5).to(DEVICE)
        self.gru = GRU_MLP_Classifier_XPU(input_dim=FEATURE_DIM, hidden_dim=128, num_classes=NUM_CLASSES).to(DEVICE)

        self.cnn.load_state_dict(torch.load(MODEL_CNN_PATH, map_location=DEVICE))
        self.gru.load_state_dict(torch.load(MODEL_GRU_PATH, map_location=DEVICE))

        self.cnn.eval()
        self.gru.eval()

    def _push_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        img_pil = Image.fromarray(img_resized)
        rgb_tensor = self.transform(img_pil)  # (3, 224, 224)

        refl_tensor = compute_reflection_map(img_resized, mode="LAB")  # (1, 224, 224)
        edge_tensor = compute_edge_map(img_resized, mode="canny")      # (1, 224, 224)

        full_tensor = torch.cat([rgb_tensor, refl_tensor, edge_tensor], dim=0)  # (5, 224, 224)
        self.window.append(full_tensor)

    def predict(self):
        if len(self.window) < SEQ_LEN:
            return None, None

        clip = torch.stack(list(self.window)).unsqueeze(0).to(DEVICE)  # (1, SEQ, 5, 224, 224)
        b, t, c, h, w = clip.shape
        clip = clip.view(b * t, c, h, w)
        with torch.no_grad():
            feat = self.cnn(clip)  # (B*T, feat_dim)
            feat = feat.view(b, t, -1)
            out = self.gru(feat)  # (B, num_classes)
            prob = F.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            conf = prob[0, pred].item()
            return LABEL_MAP[pred], conf

# ────────────────────────────── 메인 ──────────────────────────────
# ───── 메인 함수 일부 수정 ─────
def main(video_path):
    global cap, cur_frame  # for trackbar callback
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    infer = RoadInferencer(transform)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_frame = 0
    playing = True
    infer_mode = False

    print("❚❚ 스페이스: 재생/일시정지 | A/D: ±1프레임 | S/E: 구간점 이동")
    print("R: 구간 추론 활성 | Q: 종료")

    cv2.namedWindow("Video")
    cv2.createTrackbar("Seek", "Video", 0, total_frames-1, on_trackbar)

    while cap.isOpened():
        if playing or infer_mode:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ret, frame = cap.read()
            if not ret:
                break
            cur_frame += 1
            cv2.setTrackbarPos("Seek", "Video", cur_frame)

            infer._push_frame(frame)
            label, conf = infer.predict()

            # 시각화
            vis = frame.copy()
            if label:
                text = f"{label} ({conf:.2f})"
                color = (0, 255, 0)
                cv2.putText(vis, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

            vis = cv2.resize(vis, (960, 540))
            cv2.imshow("Video", vis)

        key = cv2.waitKey(30)
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            cur_frame = max(cur_frame - 1, 0)
        elif key == ord("d"):
            cur_frame = min(cur_frame + 1, total_frames - 1)
        elif key == ord("r"):
            infer_mode = not infer_mode

    cap.release()
    cv2.destroyAllWindows()

# ────────────────────────────── CLI 실행 ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Input video path")
    args = parser.parse_args()
    main(args.video)
