#CNN feature extractor(convolution + pooling + fully connected layers)

import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        # CNN 계층 정의: 3개의 Conv + ReLU + MaxPool 블록
        # 입력 채널 3(RGB), 출력 채널 16 → 32 → 64로 점진적 증가
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # 첫번째 합성곱: 입력 3채널 → 16채널, 3x3 필터, 패딩 1(출력 크기 유지)
            nn.ReLU(),                        # 비선형 활성화 함수
            nn.MaxPool2d(2),                  # 2x2 맥스풀링 (공간 크기 절반)

            nn.Conv2d(16, 32, 3, padding=1),  # 두번째 합성곱: 16채널 → 32채널
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 2x2 맥스풀링

            nn.Conv2d(32, 64, 3, padding=1),  # 세번째 합성곱: 32채널 → 64채널
            nn.ReLU(),
            nn.MaxPool2d(2)                   # 2x2 맥스풀링
        )
        # 마지막을 일차원 벡터(특징벡터)로 변환하는 완전연결 계층
        # 입력: 64채널 × 28 × 28 (224x224 이미지가 3번 풀링 거치면 28x28로 감소)
        # 출력: feature_dim(기본 128) 차원의 벡터
        self.fc = nn.Linear(64 * 28 * 28, feature_dim)

    def forward(self, x):
        # x: (batch_size, 3, 224, 224) 형태 이미지 입력
        x = self.conv_layers(x)            # CNN 계층 통과: (batch_size, 64, 28, 28)
        x = x.view(x.size(0), -1)          # 이미지 텐서를 1차원 벡터로 평탄화 (배치, 64*28*28)
        x = self.fc(x)                     # 완전연결계층 통과 (배치, feature_dim)
        return x                           # 출력은 각 이미지별 특징벡터
