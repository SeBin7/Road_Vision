import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class WindowedDataset(Dataset):
    def __init__(self, video_label_list, seq_len=5, stride=2, transform=None, class_map=None, end_sec=None):
        self.items = []
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.class_map = class_map

        print("Label Map:", self.class_map)

        for vpath, label in video_label_list:
            if not self.is_valid_video(vpath, cv2.CAP_GSTREAMER):
                print(f"❌ 무시됨 (손상 또는 프레임 없음): {vpath}")
                continue

            cap = cv2.VideoCapture(vpath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            max_frame = total_frames
            if end_sec is not None:
                max_frame = min(max_frame, int(end_sec * fps))

            for i in range(0, max_frame - seq_len + 1, stride):
                self.items.append((vpath, i, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            vpath, start_idx, label = self.items[idx]
            cap = cv2.VideoCapture(vpath, cv2.CAP_GSTREAMER)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            frames = []
            for _ in range(self.seq_len):
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"프레임 읽기 실패 at {start_idx}")
                # RGB 변환 및 PIL 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb_pil = Image.fromarray(frame_rgb)
                # RGB 채널 transform
                if self.transform:
                    img_rgb = self.transform(img_rgb_pil)  # Tensor (3, 224, 224)
                else:
                    img_rgb = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)

                # Canny edge 생성, 리사이즈, Tensor 변환
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(frame_gray, 50, 150)                # (H_orig, W_orig)
                edges_resized = cv2.resize(edges, (224, 224))         # (224, 224)
                edge_tensor = torch.from_numpy(edges_resized) \
                                .unsqueeze(0).float().div(255.0)     # Tensor (1, 224, 224)

                # 4채널 결합 (C, H, W) -> (4, 224, 224)
                img_4ch = torch.cat([img_rgb, edge_tensor], dim=0)

                frames.append(img_4ch)

            cap.release()

            if len(frames) < self.seq_len:
                raise ValueError("프레임 수 부족")

            window = torch.stack(frames)  # (seq_len, 4, 224, 224)
            lidx = self.class_map[label] if self.class_map else label
            return window, lidx

        except Exception as e:
            print(f"[❌ 오류] {vpath} index={idx} → {e}")
            return None

    def is_valid_video(self, path, check_frames=5):
        cap = cv2.VideoCapture(path, cv2.CAP_GSTREAMER)
        count = 0
        while count < check_frames:
            ret, _ = cap.read()
            if not ret:
                cap.release()
                return False
            count += 1
        cap.release()
        return True
