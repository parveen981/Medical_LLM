import sys
import os
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.xray_classifier.backend.model import load_model

# ✅ Class map
CLASS_NAMES = {
    0: "Normal",
    1: "Pneumonia"
}

# ✅ Preprocess input
def preprocess_image(image_path):
    """Loads and preprocesses an X-ray image."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")  
    return transform(image).unsqueeze(0)

# ✅ Inference function
def predict_image(model, image_path, device='cpu'):
    """Run inference on an X-ray image."""
    model.eval()
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = probabilities.max(1)

    class_name = CLASS_NAMES[predicted_class.item()]
    return class_name, confidence.item()

# ✅ Usage example
if __name__ == "__main__":
    checkpoint = "output/best_xray_model.pth"
    device = "cpu"

    model = load_model(checkpoint, device=device)

    test_image = "data/xray_dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    prediction, confidence = predict_image(model, test_image, device)

    print(f"✅ Predicted class: {prediction}")
    print(f"✅ Confidence: {confidence:.4f}")