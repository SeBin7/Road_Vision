import os, re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
from collections import defaultdict
import random

_SEQ_RE = re.compile(r"_seq(\d+)\.npz$")

# ───────────────────────────────────────────────────────────────
# ✅ 클래스별 연속 배치용 샘플러
class GroupedClassSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_to_indices = defaultdict(list)

        print("[INFO] Building label_to_indices for GroupedClassSampler...")

        # ✅ __getitem__ 없이 dataset.items에서 직접 라벨 정보 추출
        for idx, (vpath, _, label) in enumerate(dataset.items):
            if isinstance(label, str):
                label_idx = dataset.label_map[label]
            else:
                label_idx = label
            self.label_to_indices[label_idx].append(idx)

    def __iter__(self):
        all_batches = []

        for label, indices in self.label_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            batches = [
                indices[i:i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
                if len(indices[i:i + self.batch_size]) == self.batch_size
            ]
            all_batches.extend(batches)

        if self.shuffle:
            random.shuffle(all_batches)

        # 평탄화
        return iter([idx for batch in all_batches for idx in batch])

    def __len__(self):
        return len(self.dataset)

# ───────────────────────────────────────────────────────────────
def compute_reflection_map(img_rgb, mode="LAB", threshold=0.85, scale=1.0):
    if mode == "LAB":
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0:1] / 255.0
        refl = (L > threshold).astype(np.float32)
    elif mode == "HSV":
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        V = hsv[:, :, 2:3] / 255.0
        refl = (V > threshold).astype(np.float32)
    else:
        raise ValueError("Invalid refl_mode")
    return torch.from_numpy(refl * scale).permute(2, 0, 1)  # (1,H,W)

def compute_edge_map(img_rgb, mode="canny", scale=1.0):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if mode == "canny":
        edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
    elif mode == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        edge = np.sqrt(gx**2 + gy**2)
        edge = (edge / (edge.max() + 1e-6)).astype(np.float32)
    else:
        raise ValueError("Invalid edge_mode")
    return torch.from_numpy(edge * scale).unsqueeze(0)  # (1,H,W)

# ───────────────────────────────────────────────────────────────
class WindowedDataset(Dataset):
    def __init__(
        self,
        path_label_list,
        seq_len=10,
        stride=5,
        transform=None,
        class_map=None,
        label_map=None,
        use_channels=5,
        refl_mode="LAB",
        refl_threshold=0.85,
        refl_scale=0.5,
        edge_mode="canny",
        edge_scale=1.0,
        flip_block_len=0,
        flip_block_period_k=0,
        blur_prob=0.2,
        blur_ks=(3, 5),
        jitter_prob=0.3,
        jitter_b_delta=0.2,
        jitter_c_delta=0.2,
        noise_prob=0.15,
        noise_std=0.02,
        crop_prob=0.3,
        crop_scale_range=(0.8, 1.0),
        crop_aspect_range=(0.9, 1.1),
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.class_map = class_map
        self.label_map = label_map or {
            'broken': 0, 'ice_road': 1, 'normal_road': 2, 'wet_road': 3
        }
        self.idx_to_label = {v: k for k, v in self.label_map.items()}

        self.items = [(p, 0, label) for p, label in path_label_list if str(p).endswith(".npz")]
        self.use_channels = use_channels

        # 튜닝 파라미터
        self.refl_mode = refl_mode
        self.refl_threshold = refl_threshold
        self.refl_scale = refl_scale
        self.edge_mode = edge_mode
        self.edge_scale = edge_scale

        self.flip_block_len = int(flip_block_len)
        self.flip_block_period_k = int(flip_block_period_k)
        self.blur_prob = float(blur_prob)
        self.blur_ks = tuple(blur_ks)
        self.jitter_prob = float(jitter_prob)
        self.jitter_b_delta = float(jitter_b_delta)
        self.jitter_c_delta = float(jitter_c_delta)
        self.noise_prob = float(noise_prob)
        self.noise_std = float(noise_std)
        self.crop_prob = float(crop_prob)
        self.crop_scale_range = tuple(crop_scale_range)
        self.crop_aspect_range = tuple(crop_aspect_range)

        print(f"[Dataset] items={len(self.items)}, use_channels={self.use_channels}, flip_block_len={self.flip_block_len}, flip_block_period_k={self.flip_block_period_k}, crop_prob={self.crop_prob}")

    def __len__(self):
        return len(self.items)

    def _seq_index_from_name(self, npz_path: str) -> int:
        m = _SEQ_RE.search(os.path.basename(npz_path))
        return int(m.group(1)) if m else 0

    def _apply_seq_augment(self, window: torch.Tensor, npz_path_or_none=None) -> torch.Tensor:
        T, C, H, W = window.shape
        block_idx = None
        if self.flip_block_len > 0:
            sid = self._seq_index_from_name(npz_path_or_none or "")
            block_idx = sid // self.flip_block_len

        do_flip = (block_idx is not None and self.flip_block_period_k > 0 and (block_idx % self.flip_block_period_k) == 0)
        if do_flip:
            window = torch.flip(window, dims=[-1])

        if torch.rand(1).item() < self.crop_prob:
            scale = np.random.uniform(*self.crop_scale_range)
            aspect = np.random.uniform(*self.crop_aspect_range)
            crop_h = max(1, int(round(H * np.sqrt(scale / aspect))))
            crop_w = max(1, int(round(W * np.sqrt(scale * aspect))))
            crop_h = min(crop_h, H)
            crop_w = min(crop_w, W)
            top = 0 if H == crop_h else np.random.randint(0, H - crop_h + 1)
            left = 0 if W == crop_w else np.random.randint(0, W - crop_w + 1)
            window = torch.stack([TF.resized_crop(window[t], top, left, crop_h, crop_w, [H, W]) for t in range(T)], dim=0)

        if torch.rand(1).item() < self.blur_prob:
            k = int(np.random.choice(self.blur_ks))
            window = torch.stack([TF.gaussian_blur(window[t], kernel_size=k) for t in range(T)], dim=0)

        if torch.rand(1).item() < self.jitter_prob:
            b = 1.0 + (torch.rand(1).item() * 2 * self.jitter_b_delta - self.jitter_b_delta)
            c = 1.0 + (torch.rand(1).item() * 2 * self.jitter_c_delta - self.jitter_c_delta)
            window = ((window - 0.5) * c + 0.5) * b
            window = window.clamp(0, 1)

        if torch.rand(1).item() < self.noise_prob:
            window = (window + torch.randn_like(window) * self.noise_std).clamp(0, 1)

        return window

    def __getitem__(self, idx):
        vpath, _, label = self.items[idx]
        try:
            arr = np.load(str(vpath))
            arr = arr["data"] if "data" in arr else arr["arr_0"]  # (T,5,224,224)
            if arr.ndim != 4 or arr.shape[1] < 3:
                raise ValueError(f"NPZ shape error: {arr.shape}")

            T = arr.shape[0]
            new_seq = []
            for t in range(T):
                rgb = arr[t, :3]
                img_rgb = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
                refl = compute_reflection_map(img_rgb, self.refl_mode, self.refl_threshold, self.refl_scale)
                edge = compute_edge_map(img_rgb, self.edge_mode, self.edge_scale)
                frame = torch.cat([torch.from_numpy(rgb), refl, edge], dim=0)
                new_seq.append(frame)

            window = torch.stack(new_seq)  # (T,5,H,W)
            #window = self._apply_seq_augment(window, vpath)

            lidx = self.class_map[label] if self.class_map else label
            return window, lidx

        except Exception as e:
            print(f"[❌ 오류] {vpath} index={idx} → {e}")
            return None
