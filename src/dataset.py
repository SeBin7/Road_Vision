import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms

class WindowedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: './data/frames' 등
        transform: torchvision.transforms (필수)
        """
        self.transform = transform
        self.samples = []

        # wet_road 등
        class_dirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.label_map = {name: idx for idx, name in enumerate(class_dirs)}

        for class_name in class_dirs:
            class_path = os.path.join(root_dir, class_name)
            # 각 시퀀스 폴더 순회 (예: seq_00000)
            for seq_folder in sorted(os.listdir(class_path)):
                seq_path = os.path.join(class_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue
                frame_files = sorted([
                    f for f in os.listdir(seq_path)
                    if f.lower().endswith(('jpg', 'jpeg', 'png'))
                ])
                if len(frame_files) == 5:
                    frame_paths = [os.path.join(seq_path, f) for f in frame_files]
                    self.samples.append((frame_paths, self.label_map[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        images = [Image.open(f).convert("RGB") for f in frame_paths]
        if self.transform:
            images = [self.transform(img) for img in images]
        else:
            images = [transforms.ToTensor()(img) for img in images]
        images = torch.stack(images)  # (5, C, H, W)
        return images, label
