import torch.nn as nn
from transformers import OneFormerForUniversalSegmentation

from model.decoder import DeepLabHeadV3Plus


class SwinDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(SwinDeepLabV3Plus, self).__init__()

        # Swin Transformer as backbone - (pre-trained backbone)
        oneformer_swin_t = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        self.backbone = oneformer_swin_t.model.pixel_level_module.encoder

        # DeepLabv3+ as classifier
        self.classifier = DeepLabHeadV3Plus(num_classes=133)

    def forward(self, x):
        # Forward pass through Swin Transformer
        f_maps = self.backbone(x)
        features = f_maps['feature_maps']
        low_features = features[0]
        out_backbone = features[3]

        # Forward pass through DeepLabV3+
        logits_output = self.classifier(low_features, out_backbone)

        preds = logits_output.softmax(dim=1)
        preds = preds.argmax(dim=1)

        return logits_output, preds
