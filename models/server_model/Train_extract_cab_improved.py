# PyTorch 및 필요한 라이브러리 임포트
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import WindowedDataset  # 사용자 정의 데이터셋
#from CNN import CNNFeatureExtractor  # CNN 백본
from GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU  # RNN+MLP 분류기

from Mobilenet_hailo_simple import MobileNetFeatureExtractor
#from Mobilenet_hailo import MobileNetFeatureExtractor  # MobileNet 백본

import os, glob
import cv2
import random
from PIL import Image
import numpy as np

# CPU 환경 최적화: PyTorch가 사용할 CPU 스레드 수를 2개로 제한
# (여러 작업 동시 실행 시 과도한 CPU 점유 방지)
torch.set_num_threads(2)
#device = torch.device("cpu")  # 연산을 CPU에서 수행

# GPU 사용 가능 여부 확인: CUDA가 활성화된 경우 GPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #->local에선 cuda만 xpu로 전환
print(f"사용 디바이스: {device}")

# 안전한 collate 함수 정의: None이 포함된 배치를 자동으로 제거
# (데이터셋에서 오류 샘플이 있을 때 학습 중단 방지)

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
    """
    Hailo용 calibration/input_0에 개별 .npy 파일 생성
    """
    video_list, class_names = collect_labeled_videos(video_root)
    print(f"🔍 클래스: {class_names}")

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
                idx += 1
                extracted += 1
                count += 1
            cap.release()
        print(f"  {cls}: {count}장 저장")

    os.chmod(input_dir, 0o755)
    os.chmod(output_dir, 0o755)
    print(f"✅ Hailo calibration set 생성: {idx}개 파일 in {input_dir}")
    return output_dir

def train():
    video_root = '/home/team05/workspace/connection/Road_Vision/src_server_update/src_video/test_videos'
    calibration_dir = './hailo_calibration'
    calib_dir = create_hailo_calibration(video_root, calibration_dir, total_samples=128)
    calib_npy_path = os.path.join(calib_dir, 'input_0')
    video_list, class_names = collect_labeled_videos(video_root)
    label_map = {n:i for i,n in enumerate(class_names)}
    print(f"총 비디오 수: {len(video_list)}, 클래스: {label_map}")


    # 데이터 증강 및 정규화 파이프라인
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지를 224x224로 크기 통일
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 밝기/대비/채도/색조 랜덤 변화(조명 변화 대응)
        #transforms.GaussianBlur(kernel_size=3, sigma=(0.1,2.0)),  # 랜덤 블러(흐릿한 상황 대응)
        transforms.ToTensor(),  # 이미지를 [0,1] 범위의 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # 픽셀값을 ImageNet 통계로 정규화(모델 안정화)
    ])

    # ▶ WindowedDataset 생성: 시퀀스 단위 학습을 위한 커스텀 데이터셋
    dataset = WindowedDataset(
        video_list,        # [(video_path, label_name)] 리스트 → 영상과 클래스명 정보
        seq_len=5,         # 하나의 시퀀스를 구성할 프레임 개수 (8장 연속)
        stride=3,          # 슬라이딩 윈도우 간격 (ex. 4프레임 단위로 이동 → 중복 적고 효율적)
        transform=transform,     # 각 프레임에 적용할 전처리(Resize, Normalize 등)
        class_map=label_map      # {0: 'broken', 1: 'normal_road', 2: 'snow_road', 3: 'wet_road'} 등
    )

    # ▶ PyTorch DataLoader 설정: 학습을 위한 배치 구성 및 병렬 로딩
    dataloader = DataLoader(
        dataset,
        batch_size=16,         # 학습 배치 크기 (한 번에 처리할 시퀀스 수)
        shuffle=True,            # 시퀀스 순서를 매 epoch마다 섞어서 일반화 성능 향상
        num_workers=4,         # 데이터 로딩을 처리할 병렬 프로세스 수 (CPU 코어 개수 고려)
        prefetch_factor=4,       # 각 worker가 미리 로드해둘 배치 수 → 전체 prefetch는 12×4=48개
        collate_fn=safe_collate  # None 반환된 예외 샘플들을 자동으로 제거해주는 안전한 콜레이터
    )

    # 모델 초기화 (CNN 백본 + GRU-MLP 분류기)
    #cnn = CNNFeatureExtractor(feature_dim=128).to(device)  # 이미지 특징 추출 CNN
    cnn = MobileNetFeatureExtractor(feature_dim=128).to(device)  # MobileNet 백본 활용
    print(f"using {device}")

    classifier = GRU(feature_dim=128, hidden_dim=64, num_classes=len(class_names)).to(device)  # 시퀀스 분류기
    print(f"using {device}")

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류용 손실 함수
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(classifier.parameters()), lr=0.001)  # Adam 옵티마이저

    best_acc = 0.0
    patience = 3
    counter = 0

    for epoch in range(5):
        
        cnn.train()
        classifier.train()

        total_loss, correct, total = 0, 0, 0
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
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

            print(f"[E{epoch+1}] B{batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={correct/total:.4f}")

        epoch_acc = correct / total
        epoch_loss = total_loss / total
        print(f"[Epoch {epoch+1}] 평균 Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}\n")

        if epoch_acc > best_acc:
            print("🔼 모델 개선됨: 저장합니다")
            best_acc = epoch_acc
            counter = 0
            torch.save(cnn.state_dict(), "best_cnn_feature_extractor.pth")
            torch.save(classifier.state_dict(), "best_gru_mlp_classifier.pth")
            print("Saved state_dict (.pth)")
            torch.save(cnn, "best_cnn_feature_extractor.pt")
            torch.save(classifier, "best_gru_mlp_classifier.pt")
            print("Saved full model (.pt)")

            # CNN만 ONNX로 export (Hailo-8용)
            cnn.eval()
            dummy_cnn_input = torch.randn(1, 3, 224, 224).to(device)

            torch.onnx.export(
                cnn,
                dummy_cnn_input,
                "cnn_feature_extractor_mobilenetv3.onnx",
                input_names=["input"],
                output_names=["feature"],
                opset_version=11
            )

            print("✅ CNN 모델을 ONNX 형식으로 저장 완료 (cnn_feature_extractor_mobilenetv3.onnx)")
            
            # 🎯 최적화된 캘리브레이션 데이터셋 경로 출력
            print(f"🎯 Hailo 양자화 시 사용할 캘리브레이션 데이터셋:")
            print(f"   📄 .npy 파일 (권장): {calib_npy_path}")
            print(f"   📂 이미지 폴더 (백업): {calib_dir}")
            print(f"   💡 Hailo 명령어: hailo optimize --calib-set-path {calib_npy_path} model.har")

        else:
            counter += 1
            print(f"😕 정확도 개선 없음 (카운트 {counter}/{patience})")
            if counter >= patience:
                print("🛑 Early stopping 발동: 학습 즉시 종료")
                return  # 🚨 즉시 전체 학습 종료

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    #mp.set_start_method("fork", force=True)
    train()
