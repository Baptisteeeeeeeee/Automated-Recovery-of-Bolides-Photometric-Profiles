import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, efficientnet_v2_l, EfficientNet_V2_S_Weights, EfficientNet_V2_L_Weights


class EfficientNetV2_Encoder(nn.Module):
    """
    EfficientNetV2 encoder supporting 's' (small) and 'l' (large) variants
    and selective freezing of initial layers.
    """

    def __init__(self, variant='s', trainable_layers=4):
        super().__init__()

        if variant == 's':
            model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        elif variant == 'l':
            model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        features = list(model.features.children())

        self.stem = features[0]
        self.block1 = nn.Sequential(*features[1:3])
        self.block2 = features[3]
        self.block3 = features[4]
        self.block4 = features[5]

        # Freeze initial layers if trainable_layers < 5
        # children(): [stem, block1, block2, block3, block4]
        num_layers = 5
        freeze_up_to = num_layers - trainable_layers
        if freeze_up_to > 0:
            for child in list(self.children())[:freeze_up_to]:
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
    Hybrid decoder for EfficientNetV2 encoder.
    Make sure channel sizes match encoder outputs!
    """

    def __init__(self):
        super().__init__()
        self.up1 = self._up_block(160 + 128, 128)  # x5 + x4 channels
        self.up2 = self._up_block(128 + 64, 64)    # + x3 channels
        self.up3 = self._up_block(64 + 48, 32)     # + x2 channels
        self.up4 = self._up_block(32 + 24, 16)     # + x1 channels
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
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(*self.encoder(x))


if __name__ == '__main__':
    encoder = EfficientNetV2_Encoder(trainable_layers=3, variant='l')
    decoder = EfficientNetHybridDecoder()
    model = Unet(encoder, decoder)

    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)
