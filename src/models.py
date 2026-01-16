import torch.nn as nn
from torchvision import models

class AudioResNet(nn.Module):
    def __init__(self, architecture='resnet18', num_classes=2, pretrained=True):
        super(AudioResNet, self).__init__()
        
        # 1. Load Pretrained Weights (ImageNet)
        if pretrained:
            if architecture == 'resnet18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif architecture == 'resnet34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif architecture == 'resnet50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            else:
                raise ValueError("Architecture not supported.")
        else:
            weights = None

        # 2. Initialize Model
        # Standard ResNet expects input shape (Batch, 3, 224, 224)
        if architecture == 'resnet18':
            self.model = models.resnet18(weights=weights)
        elif architecture == 'resnet34':
            self.model = models.resnet34(weights=weights)
        elif architecture == 'resnet50':
            self.model = models.resnet50(weights=weights)

        # NOTE: We do NOT modify self.model.conv1 anymore.
        # We leave it as the standard 3-channel input layer.

        # 3. Modify Classifier (The Head) for 2 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)