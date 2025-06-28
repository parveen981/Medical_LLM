import sys, os
# Add both backend and project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.model_loader import load_dr_model
from backend.inference import predict_image
from backend.grad_cam import generate_gradcam
from backend.summarizer import summarize_note

# Model path relative to project root
model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'best_model_checkpoint.pth')
model = load_dr_model(model_path)

print("🧠 Testing Ophthalmology Backend...")
print("=" * 50)

# Test image path relative to project root
test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'aptos2019-blindness-detection', 'test_images', '0005cfc8afb6.png')
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'gradcam_test.jpg')

print("\n--- 👁 Inference Test ---")
try:
    class_idx, class_label = predict_image(model, test_image_path)
    print(f"✅ Predicted class: {class_idx} → {class_label}")
except Exception as e:
    print(f"❌ Inference failed: {e}")

print("\n--- 🔥 Grad-CAM Test ---")
try:
    generate_gradcam(model, test_image_path, output_path)
    print(f"✅ Grad-CAM saved to: {output_path}")
except Exception as e:
    print(f"❌ Grad-CAM failed: {e}")

print("\n--- 📝 Summarizer Test ---")
note = "Patient with blurred vision and retinal hemorrhages. Recommend ophthalmology referral."
try:
    summary = summarize_note(note)
    print(f"✅ Summary: {summary}")
except Exception as e:
    print(f"❌ Summarizer failed: {e}")

print("\n" + "=" * 50)
print("🎉 Ophthalmology Backend Test Complete!")
