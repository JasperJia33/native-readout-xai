import math
import torch
import torch.nn as nn
import torchvision


# ---------------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )
        self.store_attention = False
        self.last_spatial_attn = None

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        if self.store_attention:
            self.last_spatial_attn = sa.detach()
        x = x * sa
        return x


# ---------------------------------------------------------------------------
# ResNet-18 + CBAM
# ---------------------------------------------------------------------------
class ResNet18CBAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        pretrained = mc.get('pretrained', True)
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

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


# ---------------------------------------------------------------------------
# DenseNet-121
# ---------------------------------------------------------------------------
class DenseNet121Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        pretrained = mc.get('pretrained', True)
        weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = torchvision.models.densenet121(weights=weights)
        self.backbone.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        in_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(mc.get('dropout', 0.5)),
            nn.Linear(in_ftrs, mc['num_classes']),
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# ViT-Tiny
# ---------------------------------------------------------------------------
class ViTTiny(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']
        img_size = cfg['data']['image_size']
        patch_size = mc.get('patch_size', 8)
        embed_dim = mc.get('embed_dim', 192)
        depth = mc.get('depth', 12)
        num_heads = mc.get('num_heads', 3)
        dropout = mc.get('dropout', 0.5)
        num_classes = mc['num_classes']
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, patch_size)
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
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, N, D
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# Swin-Tiny (from scratch, 1-channel)
# ---------------------------------------------------------------------------
class SwinTiny(nn.Module):
    """Swin Transformer Tiny for single-channel input (WM-811K).

    Uses timm's swin_tiny with custom img_size and in_chans=1.
    Exposes `layers` (4 stages) for Grad-CAM on the final stage.
    """

    def __init__(self, cfg):
        super().__init__()
        import timm
        mc = cfg['model']
        img_size = cfg['data']['image_size']
        num_classes = mc['num_classes']
        dropout = mc.get('dropout', 0.5)
        window_size = mc.get('window_size', 4)

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            img_size=img_size,
            in_chans=1,
            num_classes=0,  # remove default head
            window_size=window_size,
        )
        embed_dim = self.backbone.num_features  # 768 for swin_tiny
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        # Expose final stage for Grad-CAM
        self.final_stage = self.backbone.layers[-1]

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.backbone.norm(x)  # (B, H, W, C)
        x = x.mean(dim=(1, 2))  # global average pool over spatial dims
        return self.head(x)


# ---------------------------------------------------------------------------
# Depthwise Separable CNN
# ---------------------------------------------------------------------------
class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg['model']

        def dw_block(inc, outc, stride=1):
            return nn.Sequential(
                nn.Conv2d(inc, inc, 3, stride, 1, groups=inc, bias=False),
                nn.BatchNorm2d(inc), nn.ReLU(inplace=True),
                nn.Conv2d(inc, outc, 1, bias=False),
                nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            dw_block(32, 64), dw_block(64, 128, stride=2),
            dw_block(128, 256), dw_block(256, 512, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(mc.get('dropout', 0.5)),
            nn.Linear(512, mc['num_classes']),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    'resnet18_cbam': ResNet18CBAM,
    'densenet121': DenseNet121Classifier,
    'vit_tiny': ViTTiny,
    'swin_tiny': SwinTiny,
    'depthwise_cnn': DepthwiseSeparableCNN,
}


def build_model(cfg):
    name = cfg['model']['name']
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](cfg)
