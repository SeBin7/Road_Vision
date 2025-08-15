import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class WindowedDataset(Dataset):
    def __init__(self, video_label_list, seq_len=10, stride=2,
                 transform=None, class_map=None,
                 refl_mode="LAB", edge_mode="canny", unified_transform=False):
        self.video_label_list = video_label_list
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.class_map = class_map
        self.refl_mode = refl_mode
        self.edge_mode = edge_mode
        self.unified_transform = unified_transform  # 🔹 추가

        self.items = []
        for path, label in video_label_list:
            cap = cv2.VideoCapture(path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for i in range(0, frame_count - seq_len, stride):
                self.items.append((path, i, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            vpath, start_idx, label = self.items[idx]
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            frames = []

            # ✅ 시퀀스 시작 전에 transform 파라미터 고정
            if self.unified_transform and hasattr(self.transform, "set_sequence_params"):
                self.transform.set_sequence_params()

            for i in range(self.seq_len):
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"프레임 읽기 실패 at {start_idx + i}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not self.transform:
                    raise ValueError("transform must be provided for 5ch construction")

                if self.unified_transform:
                    # ✅ 통합 방식: numpy 그대로 전달 → transform 내부에서 resize, edge, refl 처리
                    full_tensor = self.transform(frame_rgb)  # frame_rgb: (H, W, 3) numpy
                else:
                    # ✅ 기존 방식: 외부 transform은 RGB만 처리하고, 나머지는 내부에서 따로
                    img_np = cv2.resize(frame_rgb, (224, 224))           # (H, W, 3)
                    img_pil = Image.fromarray(img_np)                    # → PIL
                    rgb_tensor = self.transform(img_pil)                 # (3, 224, 224)
                    refl_tensor = self.compute_reflection_map(img_np)   # (1, 224, 224)
                    edge_tensor = self.compute_edge_map(img_np)          # (1, 224, 224)
                    full_tensor = torch.cat([rgb_tensor, refl_tensor, edge_tensor], dim=0)  # (5, 224, 224)

                frames.append(full_tensor)

            cap.release()

            if len(frames) < self.seq_len:
                raise ValueError("프레임 수 부족")

            window = torch.stack(frames)  # (seq_len, 5, 224, 224)
            lidx = self.class_map[label] if self.class_map else label
            return window, lidx

        except Exception as e:
            print(f"[❌ 오류] {vpath} index={idx} → {e}")
            return None

    def compute_reflection_map(self, img_rgb):
        if self.refl_mode == "RGB":
            gray = np.mean(img_rgb / 255.0, axis=2, keepdims=True)
            refl = (gray > 0.85).astype(np.float32)
        elif self.refl_mode == "LAB":
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0:1] / 255.0
            refl = (L > 0.85).astype(np.float32)
        elif self.refl_mode == "HSV":
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            V = hsv[:, :, 2:3] / 255.0
            refl = (V > 0.85).astype(np.float32)
        else:
            raise ValueError("Invalid refl_mode")

        return torch.from_numpy(refl).permute(2, 0, 1)  # (1, H, W)

    def compute_edge_map(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if self.edge_mode == "sobel":
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_mag = np.clip(grad_mag / grad_mag.max(), 0, 1).astype(np.float32)
        elif self.edge_mode == "canny":
            edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
            grad_mag = edge
        else:
            raise ValueError("Invalid edge_mode")

        return torch.from_numpy(grad_mag).unsqueeze(0)  # (1, H, W)
