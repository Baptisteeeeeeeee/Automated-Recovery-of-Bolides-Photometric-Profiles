import torch
import torch.nn as nn
from torchvision.models import *

class EfficientNetV2S_Encoder(nn.Module):
    """
    EfficientNetV2-S encoder with support for trainable layers.
    """

    def __init__(self,variant='s',trainable_layers=4):
        super().__init__()

        if variant == 's':
            model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        elif variant == 'm':
            model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        elif variant == 'l':
            model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Invalid variant. Choose from 's', 'm', or 'l'.")

        features = list(model.features.children())

        self.stem = features[0]
        self.block1 = nn.Sequential(*features[1:3])
        self.block2 = features[3]
        self.block3 = features[4]
        self.block4 = features[5]

        if trainable_layers < 5:
            for child in list(self.children())[:5 - trainable_layers]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)
        x4 = self.block3(x3)
        x5 = self.block4(x4)
        return x1, x2, x3, x4, x5

class EfficientNetHybridDecoder(nn.Module):
    """
    Hybrid decoder for EfficientNetV2S encoder.
    """

    def __init__(self):
        super().__init__()
        self.up1 = self._up_block(160 + 128, 128)
        self.up2 = self._up_block(128 + 64, 64)
        self.up3 = self._up_block(64 + 48, 32)
        self.up4 = self._up_block(32 + 24, 16)
        self.final = nn.Conv2d(16, 1, 1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(torch.cat([x5, x4], dim=1))
        x = self.up2(torch.cat([x, x3], dim=1))
        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        return self.final(x)

class EfficientNetHybridDecoderM(nn.Module):
    """
    Hybrid decoder for EfficientNetV2S encoder.
    """

    def __init__(self):
        super().__init__()
        self.up1 = self._up_block(160 + 176, 128)
        self.up2 = self._up_block(128 + 80, 64)
        self.up3 = self._up_block(64 + 48, 32)
        self.up4 = self._up_block(32 + 24, 16)
        self.final = nn.Conv2d(16, 1, 1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4, x5):

        x = self.up1(torch.cat([x5, x4], dim=1))

        x = self.up2(torch.cat([x, x3], dim=1))

        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))

        return self.final(x)

class EfficientNetHybridDecoderL(nn.Module):
    """
    Hybrid decoder for EfficientNetV2S encoder.
    """

    def __init__(self):
        super().__init__()
        self.up1 = self._up_block(224 + 192, 128)
        self.up2 = self._up_block(128 + 96, 64)
        self.up3 = self._up_block(64 + 64, 32)
        self.up4 = self._up_block(32 + 32, 16)
        self.final = nn.Conv2d(16, 1, 1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(torch.cat([x5, x4], dim=1))

        x = self.up2(torch.cat([x, x3], dim=1))

        x = self.up3(torch.cat([x, x2], dim=1))
        x = self.up4(torch.cat([x, x1], dim=1))
        return self.final(x)

class Unet(nn.Module):
    """
    Generic U-Net wrapper combining an encoder and decoder.
    """

    def __init__(self, encoder, decoder):
        super(Unet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(*self.encoder(x))
