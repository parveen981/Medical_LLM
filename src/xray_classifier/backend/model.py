import torch
import torch.nn as nn
from torchvision import models

class CustomModel(nn.Module):
    """Custom model for image classification."""
    def __init__(self, num_classes=2):
        super(CustomModel, self).__init__()
        # Load a pretrained ResNet18 model
        self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the final fully connected layer
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        # Pass input through the feature extractor
        out = self.features(x)
        return out

def custom_model(num_classes=2):
    """This is a factory function for the CustomModel.
    It's not used in the current implementation but can be useful for different configurations."""
    return CustomModel(num_classes)

def load_model(checkpoint_path=None, num_classes=2, device='cpu'):
    """Load the image classification model (ResNet18)"""
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
