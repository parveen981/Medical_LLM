import os
import torch
from backend.model import load_model
from backend.inference import predict_image
from backend.grad_cam import save_gradcam

# ✅ Paths
CHECKPOINT_PATH = "output/best_xray_model.pth"
TEST_IMAGE_PATH = "data/xray_dataset/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
OUTPUT_PATH = "output/xray_gradcam_result.png"

# ✅ Main Test
def test_xray_backend():
    """Test X-ray Backend: Inference + Grad-CAM."""
    device = "cpu"

    # 1️⃣ Load Model
    model = load_model(CHECKPOINT_PATH, device=device)

    # 2️⃣ Inference
    prediction, confidence = predict_image(model, TEST_IMAGE_PATH, device=device)
    print(f"✅ Predicted Class: {prediction}")
    print(f"✅ Confidence: {confidence:.4f}")

    # 3️⃣ Grad-CAM
    class_indices = {"Normal": 0, "Pneumonia": 1}
    save_gradcam(model, TEST_IMAGE_PATH, class_indices[prediction],
                 output_path=OUTPUT_PATH, device=device)

    # ✅ Done
    if os.path.exists(OUTPUT_PATH):
        print(f"✅ Grad-CAM saved at: {OUTPUT_PATH}")

if __name__ == "__main__":
    test_xray_backend()
