import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class WindowedDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=5, label_map=None):
        """
        Args:
            root_dir (str): 시퀀스 데이터 루트 디렉토리 (ex: ./data/frames)
            transform: 이미지에 적용할 torchvision 변환
            seq_len (int): 시퀀스 길이 (ex: 5프레임)
            label_map (dict, optional): 클래스 이름 → 라벨 인덱스 매핑
        """
        self.samples = []  # (시퀀스 경로, 라벨)
        self.transform = transform
        self.seq_len = seq_len

        # 클래스별 디렉토리 탐색 (예: wet_road, dry_road 등)
        class_dirs = sorted(os.listdir(root_dir))
        self.label_map = label_map or {name: idx for idx, name in enumerate(class_dirs)}

        for class_name in class_dirs:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for seq_path in sorted(glob.glob(os.path.join(class_path, "seq_*"))):
                self.samples.append((seq_path, self.label_map[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]
        frame_paths = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))

        # 시퀀스 길이 확인
        assert len(frame_paths) == self.seq_len, f"{seq_path} 시퀀스 길이 오류"

        # 이미지 로드
        images = [Image.open(p).convert("RGB") for p in frame_paths]

        # transform 적용
        if self.transform:
            images = [self.transform(img) for img in images]

        # (T, C, H, W)
        images = torch.stack(images)
        return images, label
