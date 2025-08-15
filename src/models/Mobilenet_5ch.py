import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

class SEBlockLiteStagesWeighted(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
        )
        self.global_scale = nn.Parameter(torch.ones(1))
        self.channel_weights = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        h, w = x.shape[2:]
        if h < 2 or w < 2:
            return x  # 입력 크기가 너무 작으면 생략
        y = self.reduce(x)
        scale = self.channel_weights.view(1, -1, 1, 1) * self.global_scale
        return y * scale

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # 입력 채널 수정
        first_conv = mobilenet.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        with torch.no_grad():
            new_first_conv.weight[:, :3] = first_conv.weight
            if in_channels > 3:
                mean = first_conv.weight[:, :1].mean(dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_first_conv.weight[:, i:i+1] = mean

        mobilenet.features[0][0] = new_first_conv
        self.backbone = mobilenet.features[:13]

        # 🔒 백본 파라미터 고정
        for param in self.backbone.parameters():
            param.requires_grad = False

        last_channel = mobilenet.classifier[0].in_features
        self.se = SEBlockLiteStagesWeighted(last_channel)
        self.reduce = nn.Sequential(
            nn.Conv2d(last_channel, 128, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)  # (B, C, H, W)
        x = self.se(x)        # (B, C, H, W)
        x = self.reduce(x)    # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        return x

    def eval(self):
        # 백본까지 eval 모드로 전환되도록 확장
        super().eval()
        self.backbone.eval()
        self.se.eval()
        self.reduce.eval()
        return self
