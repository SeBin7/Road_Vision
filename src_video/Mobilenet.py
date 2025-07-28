import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        base_model = mobilenet_v3_small(weights=None)  # 최신 버전에서는 pretrained 대신 weights 사용 (None이면 랜덤, 기본은 ImageNet)
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(576, feature_dim)

    def forward(self, x, flatten=True):  # flatten 인자 추가
        x = self.backbone(x)  # (B, 576, H, W)
        x = self.pool(x)      # (B, 576, 1, 1)
        if flatten:
            x = x.view(x.size(0), -1)  # (B, 576)
            x = self.fc(x)             # (B, feature_dim)
            return x
        else:
            return x  # (B, 576, 1, 1)