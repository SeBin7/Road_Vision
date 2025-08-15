# Mobilenet_hailo_4ch.py (4채널 입력용으로 수정)
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, dropout_p=0.5):
        super().__init__()
        # 1) Load pretrained MobileNetV2 backbone
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).features

        # 2) Replace first conv for 4-channel
        orig_conv = backbone[0][0]  # Conv2d
        new_conv  = nn.Conv2d(
            in_channels=4,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3]  = orig_conv.weight
            new_conv.weight[:, 3:4] = orig_conv.weight.mean(dim=1, keepdim=True)
        backbone[0] = new_conv

        # 3) Trim off final AdaptiveAvgPool2d (global pooling)
        #    In MobileNetV2, AdaptiveAvgPool2d appears as backbone[-1]
        if isinstance(backbone[-1], nn.AdaptiveAvgPool2d):
            backbone = backbone[:-1]
        self.backbone = backbone

        # 4) Manual 2×2 pooling: 7→3→2→1
        self.pool1 = nn.AvgPool2d(2, 2)  # 7→3
        self.pool2 = nn.AvgPool2d(2, 1)  # 3→2
        self.pool3 = nn.AvgPool2d(2, 2)  # 2→1

        # 5) Projection & dropout
        last_channels = backbone[-1].out_channels  # typically 1280
        self.feature_proj = nn.Conv2d(last_channels, feature_dim, kernel_size=1)
        self.dropout      = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.backbone(x)   # → (B,C,7,7)
        x = self.pool1(x)      # → (B,C,3,3)
        x = self.pool2(x)      # → (B,C,2,2)
        x = self.pool3(x)      # → (B,C,1,1)
        x = self.feature_proj(x)

        x = x.squeeze(-1).squeeze(-1)
        x = self.dropout(x) 
        
        return x # (B, feature_dim)
