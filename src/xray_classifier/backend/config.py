import os


class XRayConfig:
    """Configuration for X-ray classification."""
    DATA_DIR = os.path.abspath("c:\\Users\\parve\\Desktop\\Project-1\\June_Pep\\Medical_AI_Assistant\\data\\xray_dataset\\chest_xray")
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    CHECKPOINT_PATH = os.path.abspath("c:\\Users\\parve\\Desktop\\Project-1\\June_Pep\\Medical_AI_Assistant\\output\\best_xray_model.pth")
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    TEST_IMAGE_PATH = os.path.abspath("c:\\Users\\parve\\Desktop\\Project-1\\June_Pep\\Medical_AI_Assistant\\data\\xray_dataset\\chest_xray\\test\\NORMAL\\NORMAL2-IM-0007-0001.jpeg")
    GRADCAM_OUTPUT_PATH = os.path.abspath("c:\\Users\\parve\\Desktop\\Project-1\\June_Pep\\Medical_AI_Assistant\\output\\gradcam_visualization.png")
    
    # Additional attributes needed for the frontend
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 2

# Create an instance for easy import
xray_config = XRayConfig()
