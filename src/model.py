import torch.nn as nn
import torchvision.models as models


class EyeClassificationModel(nn.Module):
    """ResNet50 model with the custom eye-disease classification head."""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.feature_extractor = models.resnet50(weights=None)

        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False

        feature_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.feature_extractor(x)
