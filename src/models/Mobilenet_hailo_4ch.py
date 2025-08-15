# Mobilenet_hailo_4ch.py (4채널 입력용으로 수정)
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, dropout_p=0.5):
        super().__init__()
        # 1) 사전학습된 모델의 초기 가중치 사용
        base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1) 
        
        # 2) backbone의 첫 번째 Conv2d 레이어를 4채널 입력용으로 교체
        #    original_conv = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        orig_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=4,            # RGB(3) + Canny edge(1)
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        # 3) 기존 weight를 복사 및 edge 채널 weight 초기화
        with torch.no_grad():
            # RGB 채널 weight 복사
            new_conv.weight[:, :3, :, :] = orig_conv.weight
            # Canny 채널 weight는 RGB 평균값으로 초기화
            new_conv.weight[:, 3:4, :, :] = orig_conv.weight.mean(dim=1, keepdim=True)
        
        # 4) backbone에 새 conv 반영
        #    features[0] = Sequential(new_conv, BatchNorm2d(16), Hardswish())
        new_features = base_model.features
        new_features[0] = nn.Sequential(
            new_conv,
            new_features[0][1],  # BatchNorm2d(16)
            new_features[0][2],  # Hardswish()
        )
        self.backbone = new_features
        
        # 5) 나머지 풀링·projection 정의 (변경 없음)
        self.pool1 = nn.AvgPool2d(2, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(2, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(2, stride=1, padding=0)
        self.pool4 = nn.AvgPool2d(2, stride=1, padding=0)
        self.pool5 = nn.AvgPool2d(2, stride=1, padding=0)
        self.pool6 = nn.AvgPool2d(2, stride=1, padding=0)
        self.feature_proj = nn.Conv2d(576, feature_dim, kernel_size=1)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # x shape: (B, 4, H, W) – 마지막 채널은 Canny edge map
        x = self.backbone(x)        # (B, 576, 7, 7)
        x = self.pool1(x)           # (B, 576, 6, 6)
        x = self.pool2(x)           # (B, 576, 5, 5)
        x = self.pool3(x)           # (B, 576, 4, 4)
        x = self.pool4(x)           # (B, 576, 3, 3)
        x = self.pool5(x)           # (B, 576, 2, 2)
        x = self.pool6(x)           # (B, 576, 1, 1)
        x = self.feature_proj(x)    # (B, feature_dim, 1, 1)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(x) 
        
        return x # (B, feature_dim)
