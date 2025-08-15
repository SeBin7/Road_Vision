#!/usr/bin/env python
# coding: utf-8
"""
Gradual-unfreeze MobileNet + GRU classifier
– 4-채널( RGB+Edge ) 비디오 윈도우 입력
– 정확한 Accuracy 계산
– ReduceLROnPlateau + EarlyStopping(여유 10 epoch)
"""

import os, glob, random, cv2, numpy as np
from PIL import Image
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as F

import intel_extension_for_pytorch as ipex

from datasets.Dataset_4ch_npz import NPZSequenceBuilder, NPZWindowedDataset

from models.Mobilenet_hailo_4ch import MobileNetFeatureExtractor
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

from sklearn.metrics import classification_report, confusion_matrix


os.environ['IPEX_VERBOSE'] = '0'

import logging
# IPEX 로거 이름 추적 (보통 'IPEX' 혹은 '_logger' 등)
logger = logging.getLogger("IPEX")
# INFO 레벨 이상의 메시지를 출력하지 않도록 WARNING으로 설정
logger.setLevel(logging.WARNING)



# ──────────────────────────────────────────
# 하이퍼파라미터
SEQ_LEN          = 10
STRIDE           = 5 #5
BATCH_SIZE       = 16 #32, 64
BASE_LR_HEAD     = 1e-3      # 분류기 / 백본 후반부
BASE_LR_BACKBONE = 1e-4      # 백본 앞부분 (미세조정)
EPOCHS           = 10
EARLY_PATIENCE   = 3        # 개선 없을 때 조기 종료
VAL_SPLIT        = 0.2
NUM_WORKERS      = 4
# ──────────────────────────────────────────

torch.set_num_threads(4)
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"📟 Device: {device}")

# ──────────────────────────────────────────
# 유틸
def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def collect_labeled_videos(root_dir):
    video_list, class_names = [], sorted(os.listdir(root_dir))
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir): 
            continue
        for vf in glob.glob(os.path.join(cls_dir, '*.mp4')):
            video_list.append((vf, cls))
    return video_list, class_names

def create_npz_dataset_from_videos(video_root, npz_output_dir, seq_len=SEQ_LEN, stride=STRIDE):
    """비디오 폴더에서 .npz 데이터셋 생성"""
    
    # 비디오 파일 수집
    video_list = []
    class_names = []
    
    for cls in sorted(os.listdir(video_root)):
        cls_dir = os.path.join(video_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        class_names.append(cls)
        for vf in glob.glob(os.path.join(cls_dir, '*.mp4')):
            video_list.append((vf, cls))
    
    # NPZ 시퀀스 빌더로 변환
    builder = NPZSequenceBuilder(
        video_label_list=video_list,
        output_dir=npz_output_dir,
        seq_len=seq_len,
        stride=stride,
        frame_size=(224, 224)
    )
    
    total_sequences = builder.build()
    
    # 클래스 맵 생성
    class_map = {name: idx for idx, name in enumerate(class_names)}

    return total_sequences, class_map


def check_npz_labels(npz_dir):
    all_labels = set()
    npz_files = glob.glob(f"{npz_dir}/*.npz")
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        label = str(data['label'].item()) if isinstance(data['label'], np.ndarray) else str(data['label'])
        all_labels.add(label)
    
    print("모든 라벨:", sorted(all_labels))
    class_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
    return class_map

# NPZ 디렉토리 확인
npz_dir = "/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/src_video/npz_sequences"  # 실제 경로로 변경
class_map = check_npz_labels(npz_dir)
print("사용할 class_map:", class_map)


def create_hailo_calibration(video_root, output_dir, total_samples=128):
    video_list, class_names = collect_labeled_videos(video_root)
    per_class = total_samples // len(class_names)
    class_videos = {}
    for vp, cls in video_list:
        class_videos.setdefault(cls, []).append(vp)
    input_dir = os.path.join(output_dir, 'input_0')
    os.makedirs(input_dir, exist_ok=True)
    idx = 0

    for cls, vids in class_videos.items():
        samples_per_video = max(1, per_class // len(vids))
        count = 0

        for vp in vids:
            if count >= per_class: break

            cap = cv2.VideoCapture(vp)

            if not cap.isOpened():
                print(f"Failed to open pipeline for {vp}")
                continue
            
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total == 0: continue
            step, extracted = max(1, total//samples_per_video), 0
            cap = cv2.VideoCapture(vp)

            for f in range(0, total, step):
                if extracted>=samples_per_video or count>=per_class: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()

                if not ret: continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224,224))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray,50,150)
                edges = cv2.resize(edges,(224,224))
                frame4 = np.concatenate([rgb, edges[:,:,None]],axis=2).astype(np.uint8)
                path = os.path.join(input_dir, f"sample_{idx:05d}.npy")
                np.save(path, frame4); os.chmod(path,0o644)

                idx += 1
                extracted += 1
                count += 1
            cap.release()
        print(f"  {cls}: {count}장 저장")
    os.chmod(input_dir,0o755); os.chmod(output_dir,0o755)
    print(f"✅ Hailo calibration set 생성: {idx}개 파일 in {input_dir}")
    return output_dir


class SequenceConsistentFlip:
    """모든 프레임에 동일한 좌우 반전 시퀀스 증강"""
    def __init__(self, p=0.5):
        self.p = p
        self.do_flip = None
    def __call__(self, frames):
        if self.do_flip is None:
            self.do_flip = random.random() < self.p
        return [F.hflip(f) for f in frames] if self.do_flip else frames
    def reset(self):
        self.do_flip = None


def calculate_accuracy(outputs, labels):
    """다중 클래스 정확도"""
    _, preds = torch.max(outputs, 1)
    correct  = (preds == labels).sum().item()
    return correct / labels.size(0)


# ──────────────────────────────────────────
def train():
    video_root = '/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/Train_videos'
    npz_dir = '/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/src_video/npz_sequences'  # NPZ 파일이 저장될 디렉토리
    calibration_dir = './hailo_calibration'
    
     # 1단계: NPZ 데이터셋 준비 (없으면 생성, 있으면 재사용)
    if not os.path.exists(npz_dir) or len(os.listdir(npz_dir)) == 0:
        print("🔄 NPZ 데이터셋 생성 중...")
        # 비디오 파일 수집
        video_list, class_names = collect_labeled_videos(video_root)
        
        # NPZ 시퀀스 빌더로 변환
        builder = NPZSequenceBuilder(
            video_label_list=video_list,
            output_dir=npz_dir,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            frame_size=(224, 224)
        )
        total_sequences = builder.build()
        print(f"✅ {total_sequences}개 시퀀스 생성 완료")
    else:
        print("📁 기존 NPZ 데이터셋 사용")


    # 2단계: 클래스 맵 생성
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    class_names = sorted(set([os.path.basename(f).split('_')[0] for f in npz_files]))
    label_map = check_npz_labels(npz_dir)

    print(f"총 NPZ 파일 수: {len(npz_files)}, 클래스: {label_map}")

    # 3단계: Hailo calibration 생성 (기존 코드 유지)
    video_list, _ = collect_labeled_videos(video_root)  # calibration용
    calib_dir = create_hailo_calibration(video_root, calibration_dir)
    calib_npy_path = os.path.join(calib_dir, 'input_0')

    # 4단계: 클래스별 샘플 수 집계 (NPZ 기반)
    class_counts = {name: 0 for name in class_names}
    for npz_file in npz_files:
        cls_name = os.path.basename(npz_file).split('_')[0]
        class_counts[cls_name] += 1
    
    counts = [class_counts[name] for name in class_names]
    weights = [sum(counts) / c if c > 0 else 0.0 for c in counts]
    weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    print(f"클래스별 샘플 수: {class_counts}")

    

    # 5단계: Transform 정의
    seq_flip = SequenceConsistentFlip(p=0.5)
    tr_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor()
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    from torch.utils.data import Subset

    # 7단계: NPZ Dataset 생성 (디렉토리만 전달)
    # full_ds = NPZWindowedDataset(
    #     npz_dir=npz_dir,
    #     class_map=label_map,
    #     transform=tr_tf,
    #     pre_sequence_transform=seq_flip
    # )

    # 1) 학습용 Dataset: tr_tf 적용
    train_full_ds = NPZWindowedDataset(
        npz_dir=npz_dir,
        class_map=label_map,
        transform=tr_tf,
        pre_sequence_transform=seq_flip
    )

    # 2) 검증용 Dataset: val_tf 적용
    val_full_ds = NPZWindowedDataset(
        npz_dir=npz_dir,
        class_map=label_map,
        transform=val_tf,
        pre_sequence_transform=None  # sequence-level 증강은 학습에만
    )


    # train/val 인덱스 계산
    dataset_size = len(train_full_ds)  # or len(val_full_ds), 동일
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(VAL_SPLIT * dataset_size)
    val_indices   = indices[:split]
    train_indices = indices[split:]

    train_ds = Subset(train_full_ds, train_indices)
    val_ds   = Subset(val_full_ds, val_indices)


     # ─── 클래스 균형을 위한 WeightedRandomSampler 설정 ────────────────

    from collections import Counter
    # ─── 대안: NPZ 파일에서 직접 레이블 읽기 ────────────────
    # 1) train_ds의 실제 레이블 목록 수집
    train_labels = []
    for i in train_indices:
        npz_file = train_ds.dataset.npz_files[i]
        data = np.load(npz_file)
        label = str(data['label'].item()) if isinstance(data['label'], np.ndarray) else str(data['label'])
        train_labels.append(label)

    print("=== 디버깅 정보 ===")
    print("train_labels 샘플:", train_labels[:10])
    print("train_labels 고유값:", set(train_labels))
    print("label_map 키:", label_map.keys())
    print("==================")

    # 2) 레이블을 클래스 인덱스로 변환
    label_idx_list = [label_map[label] for label in train_labels]
    counts = Counter(label_idx_list)

    # 3) 클래스별 가중치 = 전체 train 샘플 수 / 클래스 샘플 수
    num_train = len(train_ds)
    class_weights = {cls: num_train / cnt for cls, cnt in counts.items()}

    # 4) 각 샘플별 가중치 리스트
    sample_weights = [class_weights[label_idx] for label_idx in label_idx_list]  # 수정된 부분

    # 5) WeightedRandomSampler 생성
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_train,
        replacement=True
    )

    # 8단계: DataLoader 생성
    # DataLoader 생성
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=4,
        collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=4,
        collate_fn=safe_collate
    )


    # 모델 --------------------------------------------------------
    cnn = MobileNetFeatureExtractor(feature_dim=128).to(device)
    cls = GRU(feature_dim=128, hidden_dim=64,
              num_classes=len(class_names)).to(device)

    # Gradual unfreeze : 백본 앞부분만 동결 ----------------------
    backbone_children = list(cnn.backbone.children())
    freeze_to = -2                 # 마지막 두 블록만 학습
    for layer in backbone_children[:freeze_to]:
        for p in layer.parameters():
            p.requires_grad = False

    # 옵티마이저 (파라미터 그룹별 LR) ----------------------------
    backbone_trainable = [p for p in cnn.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_trainable, 'lr': BASE_LR_BACKBONE},
            {'params': cls.parameters(),   'lr': BASE_LR_HEAD},
        ],
         weight_decay=1e-4
    )

    # CrossEntropyLoss에 클래스별 가중치 전달
    criterion  = nn.CrossEntropyLoss(weight=weight_tensor)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3
    )

    best_val_acc = 0.0
    no_improve   = 0

    # 학습 --------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        # ───────── TRAIN ─────────
        cnn.train()
        cls.train()
        seq_flip.reset()

        epoch_loss, epoch_correct, epoch_total = 0, 0, 0

        for bidx, batch in enumerate(train_loader):
            if batch is None: 
                continue
            windows, labels = batch
            b, t, c, h, w   = windows.shape
            inp  = windows.view(b*t, c, h, w).to(device)
            lbl  = labels.to(device)

            # IPEX 최적화 (in-place once per epoch)
            cnn, optimizer = ipex.optimize(model=cnn, optimizer=optimizer,
                                           dtype=torch.float32, inplace=True)
            cls, optimizer = ipex.optimize(model=cls, optimizer=optimizer,
                                           dtype=torch.float32, inplace=True)
            feats = cnn(inp).view(b, t, -1)
            out   = cls(feats)

            loss  = criterion(out, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc   = calculate_accuracy(out, lbl)

            epoch_loss    += loss.item() * b
            epoch_correct += acc * b
            epoch_total   += b

            if bidx % 20 == 0:
                print(f"[E{epoch:02d}] Train {bidx}/{len(train_loader)} | "
                      f"Loss {loss.item():.4f} | Acc {acc:.3f}")

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total
        print(f"▶ Epoch {epoch} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.3f}")

        # ───────── VALIDATION ─────────

        all_preds = []
        all_labels = []

        cnn.eval()
        cls.eval()

        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for vb, batch in enumerate(val_loader):
                if batch is None: 
                    continue
                windows, labels = batch
                b, t, c, h, w = windows.shape
                inp = windows.view(b*t, c, h, w).to(device)
                lbl = labels.to(device)

                feats = cnn(inp).view(b, t, -1)
                out   = cls(feats) #logits

                 # softmax → 예측 클래스
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(lbl.cpu().tolist())


                loss = criterion(out, lbl)
                acc  = calculate_accuracy(out, lbl)

                val_loss    += loss.item() * b
                val_correct += acc * b
                val_total   += b

        val_loss /= val_total
        val_acc   = val_correct / val_total
        print(f"✅ Epoch {epoch} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.3f}")

        # 스케줄러 & EarlyStopping ------------------------------
        scheduler.step(val_loss)

        # validation 루프 이후의 리포트 부분을 다음으로 교체:# 에폭별 Classification Report - 안전한 버전
        try:
            print("Classification Report:")
            print(classification_report(all_labels,
                                        all_preds,
                                        labels=list(range(len(class_names))),
                                        target_names=class_names,
                                        digits=4,
                                        zero_division=0))
        except ValueError as e:
            print(f"Classification report error: {e}")
            # 실제 나타난 클래스만으로 리포트 생성
            unique_labels = sorted(list(set(all_labels + all_preds)))
            if unique_labels:
                filtered_names = [class_names[i] for i in unique_labels if i < len(class_names)]
                print(classification_report(all_labels,
                                            all_preds,
                                            labels=unique_labels,
                                            target_names=filtered_names,
                                            digits=4))

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
        print("Confusion Matrix:")
        print(cm)

        # (선택) 시각화
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Epoch {epoch} Confusion Matrix')
        plt.show()


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            print("💾 New best! Saving checkpoints …")
            torch.save(cnn.state_dict(), "cnn_feature_extractor_4ch_val.pth")
            torch.save(cls.state_dict(), "gru_mlp_classifier_4ch_val.pth")
            torch.save(cnn, "cnn_feature_extractor_4ch_val.pt")
            torch.save(cls, "gru_mlp_classifier_4ch_val.pt")

            # ONNX export
            dummy = torch.randn(1,4,224,224).to(device)
            torch.onnx.export(cnn, dummy,
                              "cnn_feature_extractor_4ch_val.onnx",
                              input_names=["input"], output_names=["feature"],
                              opset_version=11)
            print("✅ ONNX 저장 완료")
            print("🎯 Calibration set:", calib_npy_path)

        else:
            no_improve += 1
            print(f"😕 No improvement ({no_improve}/{EARLY_PATIENCE})")

        if no_improve >= EARLY_PATIENCE:
            print("🛑 Early stopping triggered.")
            break


# ──────────────────────────────────────────
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train()
