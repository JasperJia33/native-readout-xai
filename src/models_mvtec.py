"""MVTec AD model variants — 3-channel RGB input, binary classification.

Reuses the same architectures from src/models.py but with in_channels=3
and num_classes=2. Registered separately to avoid polluting the WM-811K registry.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

from src.models import CBAM


class ResNet18CBAM_RGB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        pretrained = mc.get('pretrained', True)
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        # Keep conv1 as 3-channel (ImageNet pretrained)

        r = mc.get('cbam_reduction', 16)
        k = mc.get('cbam_kernel_size', 7)
        self.cbam1 = CBAM(64, r, k)
        self.cbam2 = CBAM(128, r, k)
        self.cbam3 = CBAM(256, r, k)
        self.cbam4 = CBAM(512, r, k)

        self.backbone.fc = nn.Sequential(
            nn.Dropout(mc.get('dropout', 0.5)),
            nn.Linear(self.backbone.fc.in_features, mc['num_classes']),
        )

    def forward(self, x):
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        x = self.cbam1(self.backbone.layer1(x))
        x = self.cbam2(self.backbone.layer2(x))
        x = self.cbam3(self.backbone.layer3(x))
        x = self.cbam4(self.backbone.layer4(x))
        x = self.backbone.avgpool(x)
        return self.backbone.fc(torch.flatten(x, 1))


class ViTTiny_RGB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        img_size = cfg['data']['image_size']
        patch_size = mc.get('patch_size', 16)
        embed_dim = mc.get('embed_dim', 192)
        depth = mc.get('depth', 12)
        num_heads = mc.get('num_heads', 3)
        dropout = mc.get('dropout', 0.5)
        num_classes = mc['num_classes']
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        return self.head(x[:, 0])


class DenseNet121_RGB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        pretrained = mc.get('pretrained', True)
        weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.densenet121(weights=weights)
        # Keep conv0 as 3-channel (ImageNet pretrained)
        in_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(mc.get('dropout', 0.5)),
            nn.Linear(in_ftrs, mc['num_classes']),
        )

    def forward(self, x):
        return self.backbone(x)


class ViTTinyPretrained_RGB(nn.Module):
    """DeiT-Tiny (ImageNet-pretrained) adapted for arbitrary image sizes."""

    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        img_size = cfg['data']['image_size']
        num_classes = mc['num_classes']
        dropout = mc.get('dropout', 0.5)

        self.backbone = timm.create_model(
            'deit_tiny_patch16_224', pretrained=True, num_classes=0,
            img_size=img_size,
        )
        embed_dim = self.backbone.embed_dim  # 192
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        # Expose for attention rollout
        self.encoder = self.backbone.blocks
        self.cls_token = self.backbone.cls_token
        self.pos_embed = self.backbone.pos_embed
        self.patch_embed = self.backbone.patch_embed

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x[:, 0]  # CLS token
        return self.head(x)


class SwinTinyPretrained_RGB(nn.Module):
    """Swin-Tiny (ImageNet-pretrained) for RGB input, binary classification."""

    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        img_size = cfg['data']['image_size']
        num_classes = mc['num_classes']
        dropout = mc.get('dropout', 0.5)
        window_size = mc.get('window_size', 7)

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            img_size=img_size,
            num_classes=0,
            window_size=window_size,
        )
        embed_dim = self.backbone.num_features  # 768
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.final_stage = self.backbone.layers[-1]

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.backbone.norm(x)  # (B, H, W, C)
        x = x.mean(dim=(1, 2))
        return self.head(x)


MVTEC_MODEL_REGISTRY = {
    'resnet18_cbam_rgb': ResNet18CBAM_RGB,
    'densenet121_rgb': DenseNet121_RGB,
    'vit_tiny_rgb': ViTTiny_RGB,
    'vit_tiny_pretrained_rgb': ViTTinyPretrained_RGB,
    'swin_tiny_pretrained_rgb': SwinTinyPretrained_RGB,
}


def build_mvtec_model(cfg):
    name = cfg['model']['name']
    if name not in MVTEC_MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MVTEC_MODEL_REGISTRY.keys())}")
    return MVTEC_MODEL_REGISTRY[name](cfg)
