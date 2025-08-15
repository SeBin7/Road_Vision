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


class NPZSequenceBuilder:
    """
    기존 WindowedDataset 로직을 이용해 시퀀스를 생성하고 .npz로 저장
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
        """단일 시퀀스 추출 (seq_len, H, W, 4)"""
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
                    frame_4ch = prev_frame
                else:
                    frame_4ch = np.zeros((*self.frame_size, 4), dtype=np.uint8)
            else:
                # RGB 채널
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, self.frame_size)
                
                # Edge 채널
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(frame_gray, 50, 150)
                edges = cv2.resize(edges, self.frame_size)
                
                # 4채널 결합 (H, W, 4)
                frame_4ch = np.concatenate([frame_rgb, edges[:, :, None]], axis=2).astype(np.uint8)
                prev_frame = frame_4ch
            
            sequence.append(frame_4ch)
        
        cap.release()
        
        if len(sequence) == self.seq_len:
            return np.stack(sequence, axis=0)  # (seq_len, H, W, 4)
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
            
        # # GStreamer 백엔드로 재시도
        # cap = cv2.VideoCapture(path, cv2.CAP_GSTREAMER)
        # count, ok = 0, True
        # while count < check_frames and ok:
        #     ok, _ = cap.read()
        #     count += 1
        # cap.release()
        # return ok


class NPZWindowedDataset(Dataset):
    """
    .npz 파일에서 시퀀스를 로드하는 Dataset
    """
    def __init__(self, npz_dir, class_map, transform=None, pre_sequence_transform=None):
        self.npz_dir = npz_dir
        self.class_map = class_map
        self.transform = transform
        self.pre_sequence_transform = pre_sequence_transform
        
        # .npz 파일 목록 수집
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        print(f"Found {len(self.npz_files)} npz files in {npz_dir}")
        print("Label Map:", self.class_map)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        
        # .npz 파일 로드
        # data = np.load(npz_path)
        # sequence = data['sequence']  # (seq_len, H, W, 4)
        # label = str(data['label'])   # numpy string -> python string

        data = np.load(self.npz_files[idx])
        sequence = data['sequence']         # (T,H,W,4)
        raw_label = data['label']    # 바로 정수 라벨


        # 라벨 변환
        # raw_label이 numpy bytes인 경우 디코딩
        # 1) raw_label이 numpy 0-d array라면 .item() 호출
        if isinstance(raw_label, np.ndarray):
            raw_label = raw_label.item()
        # 2) bytes 타입이면 디코딩
        if isinstance(raw_label, bytes):
            raw_label = raw_label.decode('utf-8')
        # 이제 raw_label은 str 라벨
        label_name = str(raw_label)

        
        # NumPy array를 PIL Image로 변환하여 transform 적용
        frames = []
        for i in range(sequence.shape[0]):
            frame = sequence[i]  # (H, W, 4)
            # RGB와 Edge 분리
            rgb_part = frame[:, :, :3]  # (H, W, 3)
            edge_part = frame[:, :, 3]  # (H, W)
            
            # PIL Image로 변환
            rgb_pil = Image.fromarray(rgb_part)
            edge_pil = Image.fromarray(edge_part, mode='L')
            
            # Transform 적용
            if self.transform:
                rgb_tensor = self.transform(rgb_pil)  # (3, H, W)
                edge_tensor = transforms.ToTensor()(edge_pil)  # (1, H, W)
            else:
                rgb_tensor = transforms.ToTensor()(rgb_pil)
                edge_tensor = transforms.ToTensor()(edge_pil)
            
            # 4채널로 결합
            frame_4ch = torch.cat([rgb_tensor, edge_tensor], dim=0)  # (4, H, W)
            frames.append(frame_4ch)
        
        # 시퀀스 단위 증강 (좌우반전 등)
        if self.pre_sequence_transform is not None:
            frames = self.pre_sequence_transform(frames)
        
        # 시퀀스 스택
        window = torch.stack(frames)  # (seq_len, 4, H, W)
        
        # 채널별 Normalize (RGB/ImageNet, Edge=0.5,0.5)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5], device=window.device)
        std = torch.tensor([0.229, 0.224, 0.225, 0.5], device=window.device)
        w = window.permute(1, 0, 2, 3)  # (4, seq_len, H, W)
        w = (w - mean[:, None, None, None]) / std[:, None, None, None]
        window = w.permute(1, 0, 2, 3)  # (seq_len, 4, H, W)
        
        
        lidx = self.class_map[label_name]

        return window, lidx

