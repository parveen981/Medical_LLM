import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from model import load_model
from config import XRayConfig

def train_model(data_dir, checkpoint_path=None, device='cpu',
                batch_size=16, num_epochs=10, learning_rate=1e-4):

    # ✅ Configuration
    config = XRayConfig()

    # ✅ Paths
    data_dir = config.DATA_DIR  # Use absolute path from config
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # ✅ Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ✅ Datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

    # ✅ Model, Criterion, Optimizer
    model = load_model(checkpoint_path, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    start_epoch = 0

    # ✅ If checkpoint available, resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Resumed from checkpoint (Epoch {start_epoch}) with Val Loss: {best_val_loss:.4f}")

    # ⚡️ Training Loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] Training"):
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # ⚡️ Validation Loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation [{epoch+1}/{config.NUM_EPOCHS}]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # ⚡️ Output Results
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ⚡️ Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }
            os.makedirs("output", exist_ok=True)
            checkpoint_path = "output/best_xray_model.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ New best checkpoint saved to: {checkpoint_path}")

    print("✅ Training Complete!")

if __name__ == "__main__":
    train_model(data_dir="data/xray_dataset/chest_xray", device="cpu", batch_size=16, num_epochs=10)
