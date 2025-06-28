import torch
from torchvision import transforms
from PIL import Image
from .config import CLASS_MAP
from .model_loader import load_dr_model

# -------------------------------
# Predict image function
# -------------------------------
def predict_image(model, image_path, return_label_name=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

    if return_label_name:
        return class_idx, CLASS_MAP[class_idx]
    else:
        return class_idx

# Optional: test block for direct script usage
if __name__ == "__main__":
    model = load_dr_model("best_model_checkpoint.pth")
    # Example image (update path as needed)
    image_path = "data/aptos2019-blindness-detection/test_images/0005cfc8afb6.png"
    class_idx, class_label = predict_image(model, image_path)
    print(f"Predicted class: {class_idx} → {class_label}")
