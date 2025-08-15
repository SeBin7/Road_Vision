import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

# 비디오 파일 → 유효성 체크 → 윈도우(시퀀스) 단위로 잘라서 → 샘플(윈도우/레이블) 생성 → 모델 입력

class WindowedDataset(Dataset):
    def __init__(self, video_label_list, seq_len=5, stride=2, transform=None, class_map=None, end_sec=None):
        self.items = []
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.class_map = class_map

        print("Label Map:", self.class_map)

        for vpath, label in video_label_list:
            if not self.is_valid_video(vpath):
                print(f"❌ 무시됨 (손상 또는 프레임 없음): {vpath}")
                continue

            cap = cv2.VideoCapture(vpath, cv2.CAP_GSTREAMER)
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
            for i in range(self.seq_len):
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"프레임 읽기 실패 at {start_idx + i}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            cap.release()

            if len(frames) < self.seq_len:
                raise ValueError("프레임 수 부족")

            window = torch.stack(frames)
            lidx = self.class_map[label] if self.class_map else label
            return window, lidx

        except Exception as e:
            print(f"[❌ 오류] {vpath} index={idx} → {e}")
            return None  # 핵심 변경

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
