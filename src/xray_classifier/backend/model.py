import torch
import torch.nn as nn
from torchvision import models

class XRayClassifier(nn.Module):
    """Custom Densenet121 model for X-ray classification."""
    def __init__(self, num_classes=2):
        super(XRayClassifier, self).__init__()
        # Load a pretrained Densenet121 model
        self.features = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).features
        
        # Add a pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes), # Densenet121 has 1024 output features
        )

    def forward(self, x):
        # Pass input through the feature extractor
        features = self.features(x)
        
        # Pool the features
        out = self.pool(features)
        
        # Flatten the output
        out = torch.flatten(out, 1)
        
        # Pass through the classifier
        out = self.classifier(out)
        return out

def xray_classifier(num_classes=2):
    """This is a factory function for the XRayClassifier.
    It's not used in the current implementation but can be useful for different configurations."""
    return XRayClassifier(num_classes)

def load_model(checkpoint_path=None, num_classes=2, device='cpu'):
    """Load the X-ray classification model (ResNet18)"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model = model.to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Model checkpoint loaded from {checkpoint_path}")

    return model
