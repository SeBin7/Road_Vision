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
                print(f"❌ 무시됨 (손상 또는 프레임 없음): {vpath}")
                continue
            # 기본 디코더로 프레임 수 조회
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
        vpath, start_idx, label = self.items[idx]
        # 기본 디코더로 시도
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            # GStreamer 백엔드 재시도
            cap = cv2.VideoCapture(vpath, cv2.CAP_GSTREAMER)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        prev_img = None

        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                # 실패 시 이전 프레임 사용, 없으면 검정 이미지
                if prev_img is not None:
                    img_4ch = prev_img
                else:
                    img_4ch = torch.zeros((4,224,224), dtype=torch.float32)
            else:
                # 정상 프레임 처리
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb = (self.transform(Image.fromarray(frame_rgb))
                           if self.transform
                           else torch.from_numpy(frame_rgb).permute(2,0,1).float().div(255.0))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(frame_gray, 50, 150)
                edges_resized = cv2.resize(edges, (224, 224))
                edge_tensor = torch.from_numpy(edges_resized).unsqueeze(0).float().div(255.0)
                img_4ch = torch.cat([img_rgb, edge_tensor], dim=0)
                prev_img = img_4ch

            frames.append(img_4ch)

        cap.release()

        # 시퀀스 단위 증강
        if self.pre_sequence_transform is not None:
            frames = self.pre_sequence_transform(frames)

        # 시퀀스 스택
        window = torch.stack(frames)  # (seq_len, 4, 224, 224)

        # 채널별 Normalize (RGB/ImageNet, Canny=0.5,0.5)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5], device=window.device)
        std  = torch.tensor([0.229, 0.224, 0.225, 0.5], device=window.device)
        w = window.permute(1,0,2,3)  # (4, seq_len, H, W)
        w = (w - mean[:,None,None,None]) / std[:,None,None,None]
        window = w.permute(1,0,2,3)  # (seq_len, 4, H, W)

        lidx = self.class_map[label] if self.class_map else label
        return window, lidx

    def is_valid_video(self, path, check_frames=5):
        # 기본 디코더로 체크
        cap = cv2.VideoCapture(path)
        count, ok = 0, True
        while count < check_frames and ok:
            ok, _ = cap.read()
            count += 1
        cap.release()
        if ok:
            return True
        # GStreamer 백엔드로 재시도
        cap = cv2.VideoCapture(path, cv2.CAP_GSTREAMER)
        count, ok = 0, True
        while count < check_frames and ok:
            ok, _ = cap.read()
            count += 1
        cap.release()
        return ok
