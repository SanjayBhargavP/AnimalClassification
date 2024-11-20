
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        # Load the VGG16 model with Batch Normalization
        self.model = models.vgg16_bn(weights=None)  # Load without pre-trained weights
        # the classifier's final layer to match the desired number of classes
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
