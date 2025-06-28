# Model loader for DR classifier
import torch
import torchvision.models as models

def load_dr_model(path, num_classes=5):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model
