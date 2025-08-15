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
        self.global_scale    = nn.Parameter(torch.ones(1))
        self.channel_weights = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        h, w = x.shape[2:]
        if h < 2 or w < 2:
            return x
        y = self.reduce(x)
        scale = self.channel_weights.view(1, -1, 1, 1) * self.global_scale
        return y * scale


class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=5, tune_first_conv_mode='new'):
        super().__init__()
        # Load MobileNetV3 small backbone (including hidden avgpool & classifier)
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Replace first Conv to accept in_channels (RGB+Reflection+Edge)
        first_conv = mobilenet.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # Initialize weights:
        #  - RGB channels copy pretrained weights
        #  - Reflection channel (idx 3) use average of RGB filters (approximate LAB L-channel)
        #  - Edge channel (idx 4) use Kaiming init (will learn Canny-like responses)
        with torch.no_grad():
            # Copy RGB pretrained
            new_first_conv.weight[:, :3] = first_conv.weight
            # Reflection: mean of RGB filters
            mean_rgb = first_conv.weight[:, :3].mean(dim=1, keepdim=True)
            new_first_conv.weight[:, 3:4] = mean_rgb
            # Edge: Kaiming normal init
            nn.init.kaiming_normal_(new_first_conv.weight[:, 4:5], nonlinearity='relu')
            # If bias exists, zero-init
            if new_first_conv.bias is not None:
                new_first_conv.bias.zero_()

        # Insert modified conv and take only feature layers (exclude avgpool/classifier)
        mobilenet.features[0][0] = new_first_conv
        self.backbone = mobilenet.features[:13]
        # Freeze backbone parameters (including first_conv weights update control via hook)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Configure gradient flow for first_conv:
        if tune_first_conv_mode == 'all':
            # Fine-tune all channels
            new_first_conv.weight.requires_grad = True
            if new_first_conv.bias is not None:
                new_first_conv.bias.requires_grad = True
        else:
            # Only Reflection & Edge channels learn; RGB channels frozen via hook
            new_first_conv.weight.requires_grad = True
            if new_first_conv.bias is not None:
                new_first_conv.bias.requires_grad = False
            def _mask_grad(grad):
                grad = grad.clone()
                grad[:, :3, :, :] = 0
                return grad
            new_first_conv.weight.register_hook(_mask_grad)

        # Define Hailo-friendly progressive pooling + SE
        last_channel = mobilenet.classifier[0].in_features
        self.reduce = nn.Sequential(
            # 7x7 -> 4x4
            nn.Conv2d(last_channel, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            SEBlockLiteStagesWeighted(256),
            # 4x4 -> 2x2
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            SEBlockLiteStagesWeighted(128),
            # 2x2 -> 1x1
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Extra convs for channel mixing at 1x1
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )

    def forward(self, x):
        # x: (B,5,H,W)
        x = self.backbone(x)     # (B, 576, H/8, W/8)
        x = self.reduce(x)       # (B, 128, 1, 1)
        return x.view(x.size(0), -1)  # (B, 128)

    def eval(self):
        super().eval()
        self.backbone.eval()
        self.reduce.eval()
        return self
