import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "aptos2019-blindness-detection")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGE_FOLDER = os.path.join(DATA_DIR, "train_images")

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

# Set image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 16

# Custom Dataset class
class FundusDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx]['id_code']
        label = self.data.iloc[idx]['diagnosis']
        image_path = os.path.join(self.image_dir, image_id + ".png")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations (resize, tensor conversion, normalize)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
])

# Create datasets
train_dataset = FundusDataset(train_df, IMAGE_FOLDER, transform=transform)
val_dataset = FundusDataset(val_df, IMAGE_FOLDER, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Test: check one batch
if __name__ == "__main__":
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch: {labels}")
        break

    import os
    import torch.nn as nn
    import torchvision.models as models
    import torch.optim as optim

    # 1. Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load ResNet18 model with pretrained weights
    model = models.resnet18(pretrained=True)

    # 3. Modify final fully connected layer to output 5 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model = model.to(device)

    # 4. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()            # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # How the model learns

    # Checkpoint loading logic
    checkpoint_path = 'best_model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"✅ Resumed from epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        print("🆕 Starting fresh training")

    # 5. Training loop
    EPOCHS = 5
    best_val_loss = float('inf')
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                val_loss += v_loss.item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # ✅ Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, checkpoint_path)
            print(f"💾 Best model saved at epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
