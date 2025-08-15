# PyTorch 및 필요한 라이브러리 임포트
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from Dataset_3ch_aug import WindowedDataset  # 사용자 정의 데이터셋
#from CNN import CNNFeatureExtractor  # CNN 백본
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU  # RNN+MLP 분류기
from Mobilenet_hailo_3ch import MobileNetFeatureExtractor  # MobileNet 백본

from collections import Counter
from torchvision.transforms import functional as F

import os, glob
import cv2
import random
from PIL import Image
import numpy as np

# import intel_extension_for_pytorch as ipex

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
EARLY_PATIENCE   = 1        # 개선 없을 때 조기 종료
VAL_SPLIT        = 0.2
NUM_WORKERS      = 4
# ──────────────────────────────────────────

# CPU 환경 최적화: PyTorch가 사용할 CPU 스레드 수를 2개로 제한
# (여러 작업 동시 실행 시 과도한 CPU 점유 방지)
torch.set_num_threads(2)
#device = torch.device("cpu")  # 연산을 CPU에서 수행

# GPU 사용 가능 여부 확인: CUDA가 활성화된 경우 GPU 사용
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"사용 디바이스: {device}")

# 안전한 collate 함수 정의: None이 포함된 배치를 자동으로 제거
# (데이터셋에서 오류 샘플이 있을 때 학습 중단 방지)
def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# 비디오 데이터와 클래스명 수집 함수
def collect_labeled_videos(root_dir):
    """
    videos/ 안의 클래스별 디렉토리에서 mp4 파일 수집
    예: videos/wet_road/*.mp4 → ('wet_road', video_path)
    """
    video_label_list = []
    class_names = sorted(os.listdir(root_dir))
    for cls_name in class_names:
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        video_files = glob.glob(os.path.join(cls_dir, '*.mp4'))
        for vf in video_files:
            video_label_list.append((vf, cls_name))
    return video_label_list, class_names

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def collect_labeled_videos(root_dir):
    video_label_list = []
    class_names = sorted(os.listdir(root_dir))
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir): continue
        for vf in glob.glob(os.path.join(cls_dir, '*.mp4')):
            video_label_list.append((vf, cls))
    return video_label_list, class_names

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
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release()
                continue
            step = max(1, total // samples_per_video)
            extracted = 0
            for f in range(0, total, step):
                if extracted >= samples_per_video or count >= per_class:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)).astype(np.uint8)
                path = os.path.join(input_dir, f"sample_{idx:05d}.npy")
                np.save(path, frame)
                os.chmod(path, 0o644)
                idx += 1; extracted += 1; count += 1
            cap.release()
        print(f"  {cls}: {count}장 저장")
    os.chmod(input_dir, 0o755); os.chmod(output_dir, 0o755)
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


# 학습 함수
def train():
    video_root = '/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/Train_3ch_Carla'  # 폴더 기반 학습 구조
    # 🎯 캘리브레이션 데이터셋 생성 (학습 시작 전)
    calibration_dir = './hailo_3ch_calibration'
    create_hailo_calibration(video_root, calibration_dir)
    
    video_list, class_names = collect_labeled_videos(video_root)
    label_map = {name: idx for idx, name in enumerate(sorted(class_names))}  # 클래스명→정수 라벨 매핑
    print(f"총 학습 비디오 수: {len(video_list)}")
    print(f"클래스 라벨 맵: {label_map}")
    num_classes = len(label_map)                    # 2

    # 5단계: Transform 정의
    seq_flip = SequenceConsistentFlip(p=0.5)
    tr_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # 픽셀값을 ImageNet 통계로 정규화(모델 안정화)
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # 픽셀값을 ImageNet 통계로 정규화(모델 안정화)
    ])

    from torch.utils.data import Subset


    # ▶ WindowedDataset 생성: 시퀀스 단위 학습을 위한 커스텀 데이터셋
    train_ds = WindowedDataset(
        video_list,        # [(video_path, label_name)] 리스트 → 영상과 클래스명 정보
        seq_len=10,         # 하나의 시퀀스를 구성할 프레임 개수 (8장 연속)
        stride=5,          # 슬라이딩 윈도우 간격 (ex. 4프레임 단위로 이동 → 중복 적고 효율적)
        transform=tr_tf,     # 각 프레임에 적용할 전처리(Resize, Normalize 등)
        class_map=label_map,      # {0: 'normal_road', 1: 'wet_road'} 등 ---> Carla version
        pre_sequence_transform=seq_flip
    )

    val_ds = WindowedDataset(
        video_list,        # [(video_path, label_name)] 리스트 → 영상과 클래스명 정보
        seq_len=10,         # 하나의 시퀀스를 구성할 프레임 개수 (8장 연속)
        stride=5,          # 슬라이딩 윈도우 간격 (ex. 4프레임 단위로 이동 → 중복 적고 효율적)
        transform=val_tf,     # 각 프레임에 적용할 전처리(Resize, Normalize 등)
        class_map=label_map,      # {0: 'normal_road', 1: 'wet_road'} 등
        pre_sequence_transform=None
    )

     # train/val 인덱스 계산
    dataset_size = len(train_ds)  # or len(val_full_ds), 동일
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(VAL_SPLIT * dataset_size)
    val_indices   = indices[:split]
    train_indices = indices[split:]

    train_ds = Subset(train_ds, train_indices)
    val_ds   = Subset(val_ds, val_indices)

     # ─── 클래스 균형을 위한 WeightedRandomSampler 설정 ────────────────

    # 8단계: DataLoader 생성
    # DataLoader 생성
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
              num_classes=num_classes).to(device)

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
    criterion  = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3
    )

    best_val_acc = 0.0
    no_improve   = 0

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
            torch.save(cnn.state_dict(), "cnn_feature_extractor_3ch_carla.pth")
            torch.save(cls.state_dict(), "gru_mlp_classifier_3ch_carla.pth")
            torch.save(cnn, "cnn_feature_extractor_3ch_carla.pt")
            torch.save(cls, "gru_mlp_classifier_3ch_carla.pt")

            # ONNX export
            dummy = torch.randn(1,3,224,224).to(device)
            torch.onnx.export(cnn, dummy,
                              "cnn_feature_extractor_3ch_carla.onnx",
                              input_names=["input"], output_names=["feature"],
                              opset_version=11)
            print("✅ ONNX 저장 완료")
            print("🎯 Calibration set:", calibration_dir)

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
