import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from .model import load_model
from .config import XRayConfig


# ✅ Preprocess the input X-ray
def preprocess_image(image_path, device='cpu'):
    """Load and preprocess an X-ray image."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


class GradCAM:
    """
    Grad-CAM implementation for X-ray classification visualization.
    Highlights the regions in the X-ray that the model focuses on for classification.
    """
    
    def __init__(self, model, target_layer_name='layer4'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained PyTorch model
            target_layer_name: Name of the target layer to compute gradients
        """
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            # Default to the last convolutional layer of ResNet
            target_layer = self.model.resnet.layer4
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor (1, 3, H, W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            cam: Grad-CAM heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate Grad-CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def visualize_cam(self, image_path, save_path=None, class_idx=None, alpha=0.4):
        """
        Generate and visualize Grad-CAM for an X-ray image.
        
        Args:
            image_path: Path to the input X-ray image
            save_path: Path to save the visualization (optional)
            class_idx: Target class index (if None, uses predicted class)
            alpha: Transparency of the heatmap overlay
            
        Returns:
            cam: Grad-CAM heatmap
            prediction: Model prediction
            confidence: Prediction confidence
        """
        # Load and preprocess image
        original_image = Image.open(image_path).convert('L')
        
        # Preprocessing transforms
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(original_image).unsqueeze(0)
        
        # Generate Grad-CAM
        cam, predicted_class = self.generate_cam(input_tensor, class_idx)
        
        # Get prediction confidence
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        # Resize CAM to original image size
        original_size = original_image.size
        cam_resized = cv2.resize(cam, original_size)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(original_image, cmap='gray')
        axes[2].imshow(cam_resized, cmap='jet', alpha=alpha)
        class_name = XRayConfig.CLASS_NAMES[predicted_class]
        axes[2].set_title(f'Overlay\nPredicted: {class_name}\nConfidence: {confidence:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {save_path}")
        
        plt.show()
        
        return cam_resized, predicted_class, confidence


def load_model_for_gradcam(checkpoint_path):
    """
    Load a trained model for Grad-CAM visualization.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        model: Loaded model ready for Grad-CAM
    """
    model = load_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(XRayConfig.CLASS_NAMES),
        device='cpu'
    )
    
    return model


def generate_gradcam_for_image(model_path, image_path, output_path=None):
    """
    Generate Grad-CAM visualization for a single X-ray image.
    
    Args:
        model_path: Path to the trained model checkpoint
        image_path: Path to the X-ray image
        output_path: Path to save the visualization
    """
    # Load model
    model = load_model_for_gradcam(model_path)
    
    # Create Grad-CAM instance
    grad_cam = GradCAM(model)
    
    # Generate visualization
    cam, prediction, confidence = grad_cam.visualize_cam(
        image_path=image_path,
        save_path=output_path
    )
    
    class_name = XRayConfig.CLASS_NAMES[prediction]
    print(f"Prediction: {class_name} (confidence: {confidence:.3f})")
    
    return cam, prediction, confidence


# ✅ Main Function
def save_gradcam(model, image_path, class_index, output_path="output/gradcam_xray.png", device='cpu'):
    """Run Grad-CAM and save the overlay image."""
    # Preprocess the image
    image = preprocess_image(image_path, device)

    # Forward pass
    model.eval()
    image.requires_grad_()
    outputs = model(image)

    # Backward pass
    model.zero_grad()
    target_score = outputs[0, class_index]
    target_score.backward()

    # Grad-CAM
    grad_cam = GradCAM(model)
    heatmap, _ = grad_cam.generate_cam(image, class_index)

    # ✅ Create Overlay
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply Color Map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combine Heatmap with the Original Image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # ✅ Save Result
    cv2.imwrite(output_path, overlay)
    print(f"✅ Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    import os

    # Paths (update these with actual paths)
    model_path = XRayConfig.CHECKPOINT_PATH  # Use absolute path from config
    test_image_path = XRayConfig.TEST_IMAGE_PATH  # Example test image path
    output_path = XRayConfig.GRADCAM_OUTPUT_PATH  # Example output path

    # Check if model exists
    if os.path.exists(model_path):
        try:
            generate_gradcam_for_image(
                model_path=model_path,
                image_path=test_image_path,
                output_path=output_path
            )
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            print("Make sure you have a trained model checkpoint available.")
    else:
        print(f"Model not found at {model_path}")
        print("Train a model first before generating Grad-CAM visualizations.")