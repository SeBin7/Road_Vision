import os
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from Dataset_5ch import WindowedDataset
from Mobilenet_5ch import MobileNetFeatureExtractor
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"✅ Using device: {device}")
torch.set_num_threads(4)

# numpy 기반 resize + reflection + edge 처리 포함
class To5ChTensor:
    def __init__(self, refl_mode="LAB", edge_mode="canny", augment=False):
        self.refl_mode = refl_mode
        self.edge_mode = edge_mode
        self.augment = augment
        self.flip_prob = 0.5
        self.random_flip = False

    def __call__(self, img_rgb_np):
        # Resize (numpy 기반)
        img_rgb_np = cv2.resize(img_rgb_np, (224, 224))

        if self.augment and self.random_flip:
            img_rgb_np = cv2.flip(img_rgb_np, 1)  # 좌우반전

        refl = self.compute_reflection_map(img_rgb_np)
        edge = self.compute_edge_map(img_rgb_np)

        # Normalize RGB
        rgb = img_rgb_np.astype(np.float32) / 255.0  # (H,W,3)
        refl = refl[..., np.newaxis]
        edge = edge[..., np.newaxis]

        full = np.concatenate([rgb, refl, edge], axis=2)  # (H,W,5)
        tensor = torch.from_numpy(full).permute(2, 0, 1).float()  # (5,H,W)
        return tensor

    def set_sequence_params(self):
        self.random_flip = random.random() < self.flip_prob

    def compute_reflection_map(self, img_rgb):
        if self.refl_mode == "LAB":
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0:1] / 255.0
            refl = (L > 0.85).astype(np.float32)
        else:
            raise NotImplementedError
        return refl.squeeze()

    def compute_edge_map(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        return edge
    
def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def collect_labeled_videos(root_dir):
    video_label_list = []
    class_names = sorted(os.listdir(root_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    print("Label Map:", label_map)
    for cls_name in class_names:
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir): continue
        video_files = glob.glob(os.path.join(cls_dir, '*.mp4'))
        for vf in video_files:
            video_label_list.append((vf, cls_name))
    return video_label_list, label_map

def crop_top_30_percent(img):
    w, h = img.size
    top = int(h * 0.3)
    return img.crop((0, top, w, h))

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    return fig

def train():
    root_dir = "../data/videos_trimmed"
    calib_dir = "./calib_npy"
    writer = SummaryWriter(log_dir="runs/train_logs")

    # 초기 디버그 체크
    writer.add_scalar("debug/init", 1.0, 0)
    writer.add_text("RunStatus", "Training Started", 0)

    video_label_list, label_map = collect_labeled_videos(root_dir)
    train_list, val_list = train_test_split(video_label_list, test_size=0.2, random_state=42)

    transform = To5ChTensor(
        refl_mode="LAB", 
        edge_mode="canny",
        augment=True
    )

    train_dataset = WindowedDataset(
        train_list, 
        seq_len=20,
        stride=10, 
        transform=transform, 
        class_map=label_map, 
        refl_mode="LAB", 
        edge_mode="canny",
        unified_transform=True
    )
    val_dataset = WindowedDataset(
        val_list, 
        seq_len=20, 
        stride=10, 
        transform=transform, 
        class_map=label_map, 
        refl_mode="LAB", 
        edge_mode="canny",
        unified_transform=True
    )
    # ─── 여러 시퀀스 시각화 (for debugging) ──────────────
    for i in range(50):
        imgs, _ = train_dataset[i]  # (T, 5, 224, 224)
        rgb_seq = imgs[:, :3]       # (T, 3, 224, 224)

        def unnormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1)

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        unnorm_rgb = torch.stack([unnormalize(img.clone(), mean, std) for img in rgb_seq])
        grid = make_grid(unnorm_rgb, nrow=len(rgb_seq))
        writer.add_image(f"SampleSequence/test_seq_{i}", grid, 0)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=safe_collate
    )

    feature_dim = 128
    num_classes = len(label_map)
    cnn = MobileNetFeatureExtractor(in_channels=5).to(device)
    classifier = GRU_MLP_Classifier_XPU(
        input_dim=feature_dim, 
        hidden_dim=128, 
        num_classes=num_classes,
        dropout_p=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(classifier.parameters()), lr=1e-4)

    best_acc = 0
    patience = 3
    counter = 0

    # 첫 샘플 입력 이미지 기록 (학습 시작 전에)
    try:
        sample_imgs, _ = next(iter(train_loader))
        rgb_seq = sample_imgs[0, :, :3]  # (T, 3, H, W)

        def unnormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1)

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        unnorm_rgb = torch.stack([unnormalize(img.clone(), mean, std) for img in rgb_seq])
        grid = make_grid(unnorm_rgb, nrow=len(rgb_seq))
        writer.add_image("SampleSequence/initial_rgb_seq", grid, 0)
    except:
        pass

    for epoch in range(5):
        cnn.train()
        classifier.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"[E{epoch+1}] B{batch_idx}: ❌ Skipped None batch")
                continue

            windows, labels = batch
            b, win, c, h, w = windows.size()
            windows = windows.view(b * win, c, h, w).to(device)
            labels = labels.to(device)

            features = cnn(windows)
            features = features.view(b, win, -1)
            outputs = classifier(features)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += b

            if (batch_idx + 1) % 10 == 0:
                print(f"[E{epoch+1}] B{batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={correct/total:.4f}")

        if total == 0:
            print("⚠️ 전체 학습 배치가 없음. 데이터셋 확인 필요.")
            return

        epoch_acc = correct / total
        epoch_loss = total_loss / total

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        # 샘플 시퀀스 이미지 그리드 기록 (1에폭 단위)
        try:
            sample_imgs, _ = next(iter(train_loader))
            rgb_seq = sample_imgs[0, :, :3]  # (T, 3, H, W)
            unnorm_rgb = torch.stack([unnormalize(img.clone(), mean, std) for img in rgb_seq])
            grid = make_grid(unnorm_rgb, nrow=len(rgb_seq))
            writer.add_image("SampleSequence/train_rgb_seq", grid, epoch)
        except:
            pass

        print(f"\n[Epoch {epoch+1}] 평균 Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            print("🔼 모델 개선됨: 저장합니다")
            best_acc = epoch_acc
            counter = 0
            torch.save(cnn.state_dict(), "best_cnn_feature_extractor.pth")
            torch.save(classifier.state_dict(), "best_gru_mlp_classifier.pth")
            writer.add_text("Best Update", f"Epoch {epoch+1} | Acc: {epoch_acc:.4f}", epoch)
        else:
            counter += 1
            print(f"😕 정확도 개선 없음 (카운트 {counter}/{patience})")
            if counter >= patience:
                print("🚓 Early stopping 발동: 학습 즉시 종료")
                break

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(cnn.state_dict(), f"checkpoints/cnn_epoch{epoch+1:03d}.pth")
        torch.save(classifier.state_dict(), f"checkpoints/classifier_epoch{epoch+1:03d}.pth")

        # ───── Validation ─────
        cnn.eval()
        classifier.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                if batch is None: continue
                windows, labels = batch
                b, t, c, h, w = windows.size()
                windows = windows.view(b * t, c, h, w).to(device)
                labels = labels.to(device)

                features = cnn(windows).view(b, t, -1)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * b
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += b

                if (val_batch_idx + 1) % 10 == 0:
                    print(f"[E{epoch+1}] 🔍 VAL B{val_batch_idx}/{len(val_loader)}: Loss={loss.item():.4f}, Acc: {correct:.4f}")
            

        val_acc = correct / total
        val_loss /= total

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"[Epoch {epoch+1}] ✅ Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        print("📄 ONNX 변환 중...")
        dummy_input = torch.randn(1, 5, 224, 224).to(device)
        torch.onnx.export(
            cnn,
            dummy_input,
            "./cnn_feature_extractor.onnx",
            input_names=["input"],
            output_names=["feature"],
            dynamic_axes={"input": {0: "batch_size"}, "feature": {0: "batch_size"}},
            opset_version=11
        )
        print("✅ ONNX 변환 완료: cnn_feature_extractor.onnx")

        print("📁 Calibration 샘플 저장 중...")
        os.makedirs(calib_dir, exist_ok=True)
        with torch.no_grad():
            for i in range(min(256, len(train_dataset))):
                sample, _ = train_dataset[i]
                npy = sample[0].cpu().numpy()
                save_path = os.path.join(calib_dir, f"sample_{i:03d}.npy")
                with open(save_path, "wb") as f:
                    np.save(f, npy)
        print(f"✅ 총 {i+1}개 NPY 저장 완료: {calib_dir}/sample_*.npy")

    writer.close()  # ✅ 학습 종료 시 종료

if __name__ == "__main__":
    train()
