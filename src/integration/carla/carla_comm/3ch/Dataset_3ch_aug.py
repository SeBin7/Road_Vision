import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class WindowedDataset(Dataset):
    def __init__(self, video_label_list, seq_len=5, stride=2,
                 transform=None, class_map=None, end_sec=None,
                 pre_sequence_transform=None):
        self.items = []
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.class_map = class_map
        self.pre_sequence_transform = pre_sequence_transform

        print("Label Map:", self.class_map)

        for vpath, label in video_label_list:
            if not self.is_valid_video(vpath):
                print(f"❌ Ignored (invalid video): {vpath}")
                continue
            cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            max_frame = total_frames if end_sec is None else min(total_frames, int(end_sec * fps))
            for i in range(0, max_frame - seq_len + 1, stride):
                self.items.append((vpath, i, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vpath, start_idx, label = self.items[idx]
        cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(vpath, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        prev_img = None

        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                img_3ch = prev_img if prev_img is not None else torch.zeros((3,224,224), dtype=torch.float32)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                if self.transform:
                    img_3ch = self.transform(img)  # expect transform → Tensor [3,H,W]
                else:
                    img_np = torch.from_numpy(frame_rgb).permute(2,0,1).float().div(255.0)
                    img_3ch = img_np
                prev_img = img_3ch
            frames.append(img_3ch)

        cap.release()

        if self.pre_sequence_transform is not None:
            frames = self.pre_sequence_transform(frames)

        window = torch.stack(frames)  # (seq_len, 3, 224, 224)

        # Normalize per channel (ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=window.device)
        std  = torch.tensor([0.229, 0.224, 0.225], device=window.device)
        w = window.permute(1,0,2,3)  # (3, seq_len, H, W)
        w = (w - mean[:, None, None, None]) / std[:, None, None, None]
        window = w.permute(1,0,2,3)   # (seq_len, 3, H, W)

        lidx = self.class_map[label] if self.class_map else label
        return window, lidx

    def is_valid_video(self, path, check_frames=5):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        count, ok = 0, True
        while count < check_frames and ok:
            ok, _ = cap.read()
            count += 1
        cap.release()
        if ok:
            return True
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        count, ok = 0, True
        while count < check_frames and ok:
            ok, _ = cap.read()
            count += 1
        cap.release()
        return ok
