import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm


class NPZSequenceBuilder3Ch:
    """
    3채널(RGB only) 시퀀스를 생성하고 .npz로 저장
    """
    def __init__(self, video_label_list, output_dir, seq_len=5, stride=2,
                 frame_size=(224, 224), end_sec=None):
        self.video_label_list = video_label_list
        self.output_dir = output_dir
        self.seq_len = seq_len
        self.stride = stride
        self.frame_size = frame_size
        self.end_sec = end_sec
        os.makedirs(self.output_dir, exist_ok=True)

    def build(self):
        """비디오에서 시퀀스를 추출하여 .npz 파일로 저장"""
        sequence_idx = 0
        
        for vpath, label in tqdm(self.video_label_list, desc="Processing videos"):
            if not self._is_valid_video(vpath):
                print(f"❌ 무시됨 (손상 또는 프레임 없음): {vpath}")
                continue
                
            # 기본 디코더로 프레임 수 조회
            cap = cv2.VideoCapture(vpath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            max_frame = total_frames
            if self.end_sec is not None:
                max_frame = min(max_frame, int(self.end_sec * fps))
            
            # 시퀀스 추출
            for start_idx in range(0, max_frame - self.seq_len + 1, self.stride):
                sequence = self._extract_sequence(vpath, start_idx)
                if sequence is not None:
                    # 파일명: {label}_{sequence_idx:06d}.npz
                    filename = f"{label}_{sequence_idx:06d}.npz"
                    filepath = os.path.join(self.output_dir, filename)
                    np.savez_compressed(filepath, sequence=sequence, label=label)
                    sequence_idx += 1
        
        print(f"✅ {sequence_idx}개 시퀀스를 {self.output_dir}에 저장완료")
        return sequence_idx

    def _extract_sequence(self, vpath, start_idx):
        """단일 시퀀스 추출 (seq_len, H, W, 3)"""
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            cap = cv2.VideoCapture(vpath, cv2.CAP_GSTREAMER)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        sequence = []
        prev_frame = None
        
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                # 실패 시 이전 프레임 사용, 없으면 검정 이미지
                if prev_frame is not None:
                    frame_3ch = prev_frame
                else:
                    frame_3ch = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            else:
                # RGB 채널만 사용
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_3ch = cv2.resize(frame_rgb, self.frame_size)
                prev_frame = frame_3ch
            
            sequence.append(frame_3ch)
        
        cap.release()
        
        if len(sequence) == self.seq_len:
            return np.stack(sequence, axis=0)  # (seq_len, H, W, 3)
        return None

    def _is_valid_video(self, path, check_frames=5):
        """비디오 파일 유효성 검사"""
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


class NPZWindowedDataset3Ch(Dataset):
    def __init__(self, npz_dir, class_map, transform=None, pre_sequence_transform=None):
        self.transform = transform
        self.pre_sequence_transform = pre_sequence_transform

        # (파일 경로, 클래스명) 쌍 수집
        self.samples = []
        for cls_name in sorted(os.listdir(npz_dir)):
            cls_dir = os.path.join(npz_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for path in glob.glob(os.path.join(cls_dir, "*.npz")):
                self.samples.append((path, cls_name))

        self.class_map = class_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        npz_path, cls_name = self.samples[idx]
        data = np.load(npz_path)
        first_key = data.files[0]
        sequence = data[first_key]  # (T, H, W, 3) or similar

        frames = []
        for frame_np in sequence:
            # 1) 불필요 차원 제거
            frame_np = np.squeeze(frame_np)

            # 2) 차원 재배열/복제
            if frame_np.ndim == 2:
                frame_np = np.stack([frame_np]*3, axis=2)
            elif frame_np.ndim == 3 and frame_np.shape[0] == 3:
                frame_np = frame_np.transpose(1,2,0)

            # 3) float → uint8
            if np.issubdtype(frame_np.dtype, np.floating):
                frame_np = (frame_np * 255).clip(0,255).astype(np.uint8)

            # 4) PIL 변환 → transform(PIL→Tensor)
            img = Image.fromarray(frame_np)
            tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
            frames.append(tensor)

        # 시퀀스 증강
        if self.pre_sequence_transform:
            frames = self.pre_sequence_transform(frames)

        # 스택 및 Normalize
        window = torch.stack(frames)  # (T,3,H,W)
        mean = torch.tensor([0.485,0.456,0.406], device=window.device)[:,None,None]
        std  = torch.tensor([0.229,0.224,0.225], device=window.device)[:,None,None]
        window = (window - mean) / std

        lidx = self.class_map[cls_name]
        return window, lidx