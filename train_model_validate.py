import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time


# Validate the model
def validate_model(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(dataloader)


# updated train_model function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20000, patience=500):
    model.train()
    best_model_state = None
    best_epoch = -1
    best_val_loss = float('inf')
    start_time = time.time()
    val_loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        epoch_start_time = time.time()

        # train
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        train_loss = epoch_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # save current model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1

        # Calculate and display estimated time remaining
        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        remaining_time = elapsed_time * (epochs - epoch - 1)
        print(f"Time Remaining: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec")
        # Overfitting detection: Check if val_loss has been increasing for more than `patience` epochs
        if len(val_loss_history) > patience and all(
                val_loss_history[-patience:][i] > best_val_loss for i in range(1, patience)):
            print(
                f"Early stopping at epoch {epoch + 1} due to overfitting (Val Loss increasing for {patience} consecutive epochs).")
            break
    total_training_time = time.time() - start_time
    print(f"Total Training Time: {total_training_time // 60:.0f} min {total_training_time % 60:.0f} sec")
    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, f'tower_net_epoch_{best_epoch}_is_the_best_validation_loss_is_{best_val_loss:.4f}.pth')
        print(f"best model saved in epoch: {best_epoch} , Validation Loss: {best_val_loss:.4f}")
    else:
        print("Best model not found")


# Custom Datasets
class TowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Iterate through the folder, loading image paths and offset tags
        for img_name in os.listdir(root_dir):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                base_name = os.path.splitext(img_name)[0]
                parts = base_name.split('_')
                dx = int(parts[2][1:]) if parts[2].startswith('P') else -int(parts[2][1:])
                dy = int(parts[3][1:]) if parts[3].startswith('P') else -int(parts[3][1:])
                self.data.append((img_name, dx, dy))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, dx, dy = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([dx, dy], dtype=torch.float32)
        return image, label


# Defining the Model
class TowerNet(nn.Module):
    def __init__(self):
        super(TowerNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (150, 150) -> (150, 150)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (150, 150) -> (75, 75)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (75, 75) -> (75, 75)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (75, 75) -> (37, 37)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (37, 37) -> (37, 37)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (37, 37) -> (18, 18)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Output Δx and Δy
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


# Data preprocessing and loading
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

train_dataset = TowerDataset('./train', transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TowerNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer)