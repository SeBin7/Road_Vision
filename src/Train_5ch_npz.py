import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate
from torchvision.utils import make_grid

from Dataset_5ch_npz import WindowedDataset, GroupedClassSampler
from Mobilenet_5ch_npz import MobileNetFeatureExtractor
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

def load_npz_paths(root_dir):
    files = []
    classes = sorted(os.listdir(root_dir))
    label_map = {cls: i for i, cls in enumerate(classes)}
    print("Label Map:", label_map)
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith('.npz'):
                files.append((os.path.join(cls_dir, fname), cls))
    return files, label_map

def train(    
    data_root='../data/frames_test',
    seq_len=10,
    stride=5,
    batch_size=8,
    num_workers=4,
    epochs=20,
    lr=1e-4,
    patience=5,
    calib_dir='./calib_npy'
):

    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    all_paths, label_map = load_npz_paths(data_root)
    train_list, val_list = train_test_split(all_paths, test_size=0.2, random_state=42)
       
    train_ds = WindowedDataset(
        train_list, 
        seq_len=seq_len, 
        stride=stride, 
        class_map=label_map
    )
    val_ds = WindowedDataset(
        val_list, 
        seq_len=seq_len, 
        stride=stride, 
        class_map=label_map
    )    

    print(f"[INFO] Loaded {len(train_ds)} train / {len(val_ds)} val samples")
    if len(train_ds) == 0:
        print("[❌] Train dataset is empty! Adjust seq_len or check npz")
        return
    
    #sampler = GroupedClassSampler(train_ds, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        #sampler=sampler,
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=safe_collate
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=safe_collate
    )

    cnn = MobileNetFeatureExtractor(in_channels=5, tune_first_conv_mode='new').to(device)
    classifier = GRU_MLP_Classifier_XPU(input_dim=128, hidden_dim=128, num_classes=len(label_map), dropout_p=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(classifier.parameters()), lr=lr)

    writer = SummaryWriter(log_dir='runs/train_logs')

    # ─── 여러 시퀀스 시각화 (for debugging) ──────────────
    for i in range(10):  # 50개는 많아서 10개만 표시 추천
        imgs, _ = train_ds[i]  # (T, 5, 224, 224)
        rgb_seq = imgs[:, :3]  # (T, 3, 224, 224)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def unnormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return torch.clamp(tensor, 0, 1)

        unnorm_rgb = torch.stack([unnormalize(img.clone(), mean, std) for img in rgb_seq])
        grid = make_grid(unnorm_rgb, nrow=len(rgb_seq))

        writer.add_image(f"SampleSequence/train_seq_{i}", grid, 0)

    best_acc = 0.0
    best_val_acc = 0.0
    counter = 0
    mean, std = [0.5]*3, [0.5]*3

    for epoch in range(epochs):
        cnn.train()
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        idx_to_label = {v: k for k, v in train_ds.label_map.items()}        
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"[E{epoch+1}] B{batch_idx}: ❌ Skipped None batch")
                continue

            windows, labels = batch

            if batch_idx < 5:  # ✅ 앞 몇 배치만 출력
                label_str = [idx_to_label[l.item()] for l in labels]
                print(f"[BATCH {batch_idx}] Labels: {label_str}")


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

        # Validation
        cnn.eval(); classifier.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                if batch is None: continue
                windows, labels = batch
                b, t, c, h, w = windows.size()
                windows = windows.view(b*t, c, h, w).to(device)
                labels = labels.to(device)

                features = cnn(windows).view(b, t, -1)
                outputs = classifier(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * b
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += b

                if (val_batch_idx + 1) % 10 == 0:
                    print(f"[VAL E{epoch+1}] B{val_batch_idx}/{len(val_loader)}: Loss={loss.item():.4f}, Acc={correct/total:.4f}")
        
        val_loss /= total
        val_acc = correct / total
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            print("🔼 Validation 개선됨: 저장합니다")
            best_val_acc = val_acc
            counter = 0
            torch.save(cnn.state_dict(), 'best_cnn.pth')
            torch.save(classifier.state_dict(), 'best_classifier.pth')
        else:
            counter += 1
            print(f"😕 Validation 개선 없음 (카운트 {counter}/{patience})")
            if counter >= patience:
                print("[✅] Early stopping")
                break

    writer.close()

if __name__ == '__main__':
    # 1. 데이터 체크용 코드 (주석 가능)
    train_list, _ = train_test_split(load_npz_paths('../data/frames_test')[0], test_size=0.2, random_state=42)
    tmp_ds = WindowedDataset(train_list, seq_len=10, stride=5)
    print(f"[DEBUG] Dataset length: {len(tmp_ds)}")
    for i in range(min(10, len(tmp_ds))):
        result = tmp_ds[i]
        if result is None:
            print(f"[DEBUG] Sample {i} is None")
        else:
            x, y = result
            print(f"[DEBUG] Sample {i}: x.shape={x.shape}, y={y}")

    # 2. 학습 시작
    train()