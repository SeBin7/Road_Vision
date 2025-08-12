# train.py
import os
import glob
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from Dataset_4ch import WindowedDataset              # 수정된 4채널 Dataset
from Mobilenet_hailo_4ch import MobileNetFeatureExtractor  # 4채널 CNN
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU

import intel_extension_for_pytorch as ipex

# CPU 스레드 제한
torch.set_num_threads(2)

# 디바이스 설정
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"사용 디바이스: {device}")

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
            cap = cv2.VideoCapture(vp, cv2.CAP_GSTREAMER)
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

def train():
    video_root = '/home/kyj28/workspace/Road_Vision/Road_Vision_video/Road_Vision/Train_vidoes'
    calibration_dir = './hailo_calibration'
    calib_dir = create_hailo_calibration(video_root, calibration_dir)
    calib_npy_path = os.path.join(calib_dir, 'input_0')

    video_list, class_names = collect_labeled_videos(video_root)
    label_map = {n:i for i,n in enumerate(class_names)}
    print(f"총 비디오 수: {len(video_list)}, 클래스: {label_map}")

    # 4채널 학습용 transform (RGB transform 후 Canny 채널은 Dataset에서 추가)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    dataset = WindowedDataset(
        video_list,
        seq_len=5,
        stride=3,
        transform=transform,
        class_map=label_map
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        collate_fn=safe_collate
    )

    # 모델 초기화
    cnn = MobileNetFeatureExtractor(feature_dim=128).to(device)  # in_channels=4
    classifier = GRU(feature_dim=128, hidden_dim=64,
                     num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(cnn.parameters()) + list(classifier.parameters()),
        lr=0.0001
    )

    best_acc, patience, counter = 0.0, 3, 0

    for epoch in range(5):
        cnn.train() 
        #학습 성능 향상
        cnn, optimizer = ipex.optimize(
             model=cnn,
             optimizer=optimizer,
             dtype=torch.float32,
             inplace=True
         )

        classifier.train()
        classifier, optimizer = ipex.optimize(
             model=classifier,
             optimizer=optimizer,
             dtype=torch.float32,
             inplace=True
         )
        
        total_loss, correct, total = 0, 0, 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            windows, labels = batch
            b, win, c, h, w = windows.size()
            assert c == 4, f"입력 채널 오류: 기대 4채널, 실제 {c}채널"
            windows = windows.view(b*win, c, h, w).to(device)
            labels = labels.to(device)

            features = cnn(windows)               # (b*win, feature_dim)
            features = features.view(b, win, -1)  # (b, win, feature_dim)
            outputs = classifier(features)        # (b, num_classes)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*b
            preds = torch.argmax(outputs, dim=1)
            correct += (preds==labels).sum().item()
            total += b
            print(f"[E{epoch+1}] B{batch_idx}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, Acc={correct/total:.4f}")

        epoch_acc = correct/total
        epoch_loss = total_loss/total
        print(f"[Epoch {epoch+1}] 평균 Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}\n")

        if epoch_acc > best_acc:
            print("🔼 모델 개선됨: 저장합니다")
            best_acc, counter = epoch_acc, 0
            torch.save(cnn.state_dict(), "best_cnn_feature_extractor.pth")
            torch.save(classifier.state_dict(), "best_gru_mlp_classifier.pth")
            torch.save(cnn, "best_cnn_feature_extractor.pt")
            torch.save(classifier, "best_gru_mlp_classifier.pt")

            # 4채널 ONNX export
            cnn.eval()
            dummy_input = torch.randn(1, 4, 224, 224).to(device)
            torch.onnx.export(
                cnn,
                dummy_input,
                "cnn_feature_extractor_mobilenetv3_4ch.onnx",
                input_names=["input"],
                output_names=["feature"],
                opset_version=11
            )
            print("✅ ONNX 저장 완료")

            print("🎯 Calibration set:", calib_npy_path)

        else:
            counter += 1
            print(f"😕 개선 없음 ({counter}/{patience})")
            if counter >= patience:
                print("🛑 Early stopping")
                return

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    train()
