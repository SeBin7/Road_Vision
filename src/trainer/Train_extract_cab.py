# PyTorch 및 필요한 라이브러리 임포트
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import WindowedDataset  # 사용자 정의 데이터셋
#from CNN import CNNFeatureExtractor  # CNN 백본
from models.GRU_MLP_xpu import GRU_MLP_Classifier_XPU as GRU  # RNN+MLP 분류기
from Mobilenet_hailo import MobileNetFeatureExtractor  # MobileNet 백본

import os, glob
import cv2
import random
from PIL import Image
import numpy as np

import intel_extension_for_pytorch as ipex

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

def extract_calibration_frames(video_list, output_dir, frames_per_video=15, target_size=(224, 224)):
    """
    학습용 비디오들에서 캘리브레이션용 프레임을 추출하여 저장
    
    Args:
        video_list: [(video_path, class_name), ...] 형태의 비디오 리스트
        output_dir: 캘리브레이션 이미지를 저장할 디렉토리
        frames_per_video: 각 비디오에서 추출할 프레임 수
        target_size: 이미지 크기 (width, height)
    """
    print(f"🎬 캘리브레이션 데이터셋 생성 중...")
    os.makedirs(output_dir, exist_ok=True)
    
    total_extracted = 0
    
    for video_path, class_name in video_list:
        print(f"Processing: {os.path.basename(video_path)} (class: {class_name})")
        
        # 클래스별 하위 디렉토리 생성
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 비디오를 열 수 없습니다: {video_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            continue
            
        # 랜덤하게 프레임 인덱스 선택
        frame_indices = sorted(random.sample(
            range(0, total_frames), 
            min(frames_per_video, total_frames)
        ))
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # 이미지 전처리 (OpenCV BGR → RGB 변환 및 리사이즈)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, target_size)
            
            # PIL Image로 변환하여 저장
            pil_image = Image.fromarray(frame_resized)
            
            # 파일명: {비디오명}_frame_{프레임번호}.jpg
            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            pil_image.save(filepath, quality=95)
            total_extracted += 1
            
        cap.release()
        
    print(f"✅ 캘리브레이션 데이터셋 생성 완료!")
    print(f"📊 총 {total_extracted}개 프레임을 {output_dir}에 저장했습니다.")
    return output_dir

def create_calibration_dataset_from_training_videos(video_root, calib_output_dir):
    """
    학습용 비디오 디렉토리에서 캘리브레이션 데이터셋을 생성
    """
    print("🔍 학습용 비디오에서 캘리브레이션 데이터셋 생성 중...")
    
    # 학습용 비디오 수집
    video_list, class_names = collect_labeled_videos(video_root)
    print(f"발견된 클래스: {class_names}")
    print(f"총 비디오 수: {len(video_list)}")
    
    # 캘리브레이션 프레임 추출
    calib_dir = extract_calibration_frames(
        video_list, 
        calib_output_dir, 
        frames_per_video=20,  # 각 비디오당 20장
        target_size=(224, 224)
    )
    
    return calib_dir

# 학습 함수
def train():
    video_root = '../test_videos'  # 폴더 기반 학습 구조
    
    # 🎯 캘리브레이션 데이터셋 생성 (학습 시작 전)
    calibration_dir = './calibration_dataset'
    create_calibration_dataset_from_training_videos(video_root, calibration_dir)
    
    video_list, class_names = collect_labeled_videos(video_root)
    label_map = {name: idx for idx, name in enumerate(sorted(class_names))}  # 클래스명→정수 라벨 매핑
    print(f"총 학습 비디오 수: {len(video_list)}")
    print(f"클래스 라벨 맵: {label_map}")

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
        batch_size=4,           # 학습 배치 크기 (한 번에 처리할 시퀀스 수)
        shuffle=True,            # 시퀀스 순서를 매 epoch마다 섞어서 일반화 성능 향상
        num_workers=2,          # 데이터 로딩을 처리할 병렬 프로세스 수 (CPU 코어 개수 고려)
        prefetch_factor=4,       # 각 worker가 미리 로드해둘 배치 수 → 전체 prefetch는 12×4=48개
        collate_fn=safe_collate,  # None 반환된 예외 샘플들을 자동으로 제거해주는 안전한 콜레이터
        pin_memory=True #CPU blocking 방지
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
            
            # 🎯 캘리브레이션 데이터셋 경로 출력
            print(f"🎯 Hailo 양자화 시 사용할 캘리브레이션 데이터셋: {calibration_dir}")
            print(f"   명령어 예시: hailo optimize --calib-path {calibration_dir} model.har")

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
