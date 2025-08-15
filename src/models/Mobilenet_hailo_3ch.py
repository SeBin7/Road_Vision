# Mobilenet_hailo_simple.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, dropout_p=0.5):
        super().__init__()
        base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = base_model.features
        
        # 🔥 완전히 단순화된 풀링 (BatchNorm, ReLU 제거)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 7→6
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 6→5  
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 5→4
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 4→3
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 3→2
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)  # 2→1
        
        # 최종 projection (BatchNorm 제거)
        self.feature_proj = nn.Conv2d(576, feature_dim, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.backbone(x)       # (B, 576, 7, 7)
        
        # 🔥 매우 점진적 축소 (활성화 함수 없음)
        x = self.pool1(x)          # (B, 576, 6, 6)
        x = self.pool2(x)          # (B, 576, 5, 5)
        x = self.pool3(x)          # (B, 576, 4, 4)
        x = self.pool4(x)          # (B, 576, 3, 3)
        x = self.pool5(x)          # (B, 576, 2, 2)
        x = self.pool6(x)          # (B, 576, 1, 1)
        
        x = self.feature_proj(x)   # (B, feature_dim, 1, 1)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(x) 
        
        return x # (B, feature_dim)
