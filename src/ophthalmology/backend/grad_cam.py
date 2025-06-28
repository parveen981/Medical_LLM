import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from .model_loader import load_dr_model
from .config import CLASS_MAP

# ------------------------------------------------
# ✅ Grad-CAM Generator
# ------------------------------------------------
def generate_gradcam(model, image_path, output_path, target_layer='layer4'):
    """
    Generates a Grad-CAM overlay for a given image and model.

    Args:
        model: The trained DR classification model.
        image_path: Path to the input image.
        output_path: Path where the Grad-CAM result will be saved.
        target_layer: The target layer for extracting activations and gradients.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ------------------------------------------------
    # STEP 1: Preprocess the input image
    # ------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Original Image (for overlay)
    original_img = np.array(image)
    original_img = cv2.resize(original_img, (224, 224))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------
    # STEP 2: Forward Hook Registration
    # ------------------------------------------------
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = dict(model.named_modules())[target_layer]
    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)

    # ------------------------------------------------
    # STEP 3: Forward + Backward Pass
    # ------------------------------------------------
    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    # ------------------------------------------------
    # STEP 4: Generate Grad-CAM
    # ------------------------------------------------
    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # ------------------------------------------------
    # STEP 5: Create Overlay
    # ------------------------------------------------
    heatmap = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # Draw Circle Around Max Activation
    max_loc = np.unravel_index(np.argmax(cam), cam.shape)
    center = (int(max_loc[1]), int(max_loc[0]))
    cv2.circle(overlay, center, 30, (255, 255, 255), 2)

    # ------------------------------------------------
    # STEP 6: Save Result
    # ------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)

    # ------------------------------------------------
    # STEP 7: Cleanup
    # ------------------------------------------------
    handle_fwd.remove()
    handle_bwd.remove()
    print(f"✅ Grad-CAM saved to: {output_path}")

# ------------------------------------------------
# Optional: Test Block
# ------------------------------------------------
if __name__ == "__main__":
    model = load_dr_model("best_model_checkpoint.pth")
    image_path = "data/aptos2019-blindness-detection/test_images/0005cfc8afb6.png"
    output_path = "output/gradcam_result.png"
    generate_gradcam(model, image_path, output_path)

