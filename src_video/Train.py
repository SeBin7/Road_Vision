# PyTorch 및 필요한 라이브러리 임포트
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import WindowedDataset  # 사용자 정의 데이터셋
from CNN import CNNFeatureExtractor  # CNN 백본
from GRU_MLP import GRU_MLP_Classifier  # RNN+MLP 분류기
from Mobilenet import MobileNetFeatureExtractor  # MobileNet 백본
import os, glob

# ──────────────────────────── 설정 ──────────────────────────────
# 설정: 강제로 CPU 사용하거나, 스레드 수 제한을 원할 때 사용
USE_CPU_ONLY = True
NUM_CPU_THREADS = 2  # CPU 사용 시 스레드 수 제한

if USE_CPU_ONLY:
    torch.set_num_threads(NUM_CPU_THREADS)
    device = torch.device("cpu")
    print(f"사용 디바이스: {device} (스레드 {NUM_CPU_THREADS}개)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

# ──────────────────────────── 안전한 collate 함수 ──────────────────────────────
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

# 학습 함수
def train():
    video_root = '../test_videos'  # 폴더 기반 학습 구조
    video_list, class_names = collect_labeled_videos(video_root)
    label_map = {name: idx for idx, name in enumerate(sorted(class_names))}  # 클래스명→정수 라벨 매핑
    print(f"총 학습 비디오 수: {len(video_list)}")
    print(f"클래스 라벨 맵: {label_map}")

    # 데이터 증강 및 정규화 파이프라인
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지를 224x224로 크기 통일
        transforms.RandomHorizontalFlip(p=0.7),  # 70% 확률로 좌우 반전(도로 방향 다양화)
        transforms.RandomRotation(degrees=15),  # -15~+15도 내에서 랜덤 회전(카메라 각도 다양화)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 밝기/대비/채도/색조 랜덤 변화(조명 변화 대응)
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 약간의 이동, 확대/축소, 회전(시점 다양화)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1,2.0)),  # 랜덤 블러(흐릿한 상황 대응)
        transforms.ToTensor(),  # 이미지를 [0,1] 범위의 텐서로 변환
        transforms.RandomErasing(p=0.3),  # 30% 확률로 이미지 일부를 랜덤하게 지움(가려짐, 노이즈 대응)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # 픽셀값을 ImageNet 통계로 정규화(모델 안정화)
    ])

    # 윈도우 기반 데이터셋 및 DataLoader 생성
    dataset = WindowedDataset(video_list, seq_len=5, stride=3, transform=transform, class_map=label_map)  # 시퀀스 단위 데이터셋 생성
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 한 번에 4개 윈도우(시퀀스)씩 묶어서 학습
        shuffle=True,  # 에폭마다 데이터 순서 섞기
        num_workers=2,  # 데이터 로딩에 사용할 CPU 프로세스 개수
        prefetch_factor=2,  # 각 worker가 미리 준비할 배치 수
        collate_fn=safe_collate  # 오류 샘플이 있으면 자동으로 배치에서 제거
    )

    # 모델 초기화 (CNN 백본 + GRU-MLP 분류기)
    # cnn = CNNFeatureExtractor(feature_dim=128).to(device)  # 이미지 특징 추출 CNN
    cnn = MobileNetFeatureExtractor(feature_dim=128).to(device)  # MobileNet 백본 활용
    classifier = GRU_MLP_Classifier(feature_dim=128, hidden_dim=64, num_classes=len(class_names)).to(device)  # 시퀀스 분류기

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류용 손실 함수
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(classifier.parameters()), lr=0.001)  # Adam 옵티마이저

    # 에폭 반복 학습
    for epoch in range(4):
        cnn.train(); classifier.train()  # 학습 모드 전환
        total_loss, correct, total = 0, 0, 0
        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue  # 오류 샘플 배치 건너뜀
            windows, labels = batch  # 윈도우 시퀀스, 라벨
            b, win, c, h, w = windows.size()  # 배치 크기, 윈도우 길이, 채널, 높이, 너비
            windows = windows.view(b * win, c, h, w).to(device)  # CNN 입력 형태로 변환
            labels = labels.to(device)

            features = cnn(windows, flatten=True)  # CNN 특징 추출
            features = features.view(b, win, -1)  # (batch, window, feature_dim)
            outputs = classifier(features)  # RNN+MLP 분류 결과

            loss = criterion(outputs, labels)  # 손실 계산
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += b

            if batch_idx % 10 == 0:
                print(f"[E{epoch+1}] B{batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={correct/total:.4f}")

        print(f"[Epoch {epoch+1}] 평균 Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}\n")

    # 학습된 모델 저장
    torch.save(cnn.state_dict(), 'cnn_feature_extractor.pth')
    torch.save(classifier.state_dict(), 'gru_mlp_classifier.pth')
    print("✅ 모델 저장 완료")

if __name__ == "__main__":
    train()
