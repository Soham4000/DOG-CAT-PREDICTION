"""
Run this in Colab. This is your exact architecture (16 -> 32 -> 32 conv blocks,
Dense(512) -> Dense(1)) rewritten in PyTorch instead of Keras, with one change:
Flatten() is replaced by GlobalAveragePooling. Your original Flatten+Dense(512)
combo produces an 8.67M-parameter layer (~35MB) because 200x200 input with valid
padding leaves a 23x23x32 feature map before the Dense layer. Global average
pooling collapses that to 32 values first, keeping the same Dense(512)->Dense(1)
head but cutting the model to well under 1MB.
"""

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 1. Unzip dataset ----------
if not os.path.exists('/content/DOGCAT'):
    with zipfile.ZipFile('/content/DOGCAT.zip', 'r') as zip_ref:
        zip_ref.extractall('/content')

# ---------- 2. Data loading ----------
IMG_SIZE = 200  # matches your original target_size=(200,200)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # scales pixels to [0,1], same as your rescale=1/255
])

train_dataset = datasets.ImageFolder('/content/DOGCAT/training', transform=train_transform)
val_dataset = datasets.ImageFolder('/content/DOGCAT/validation', transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print("Classes:", train_dataset.class_to_idx)  # confirm 0=cat, 1=dog (alphabetical, matches your class_indices)

# ---------- 3. Model — same conv stack as your Keras version, GAP instead of Flatten ----------
class DogCatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # replaces Flatten -> avoids the 8.67M-param blowup
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DogCatCNN().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())  # matches your optimizer='adam'

# ---------- 4. Train ----------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} - loss: {running_loss/total:.4f} - "
          f"train_acc: {train_acc:.4f} - val_acc: {val_correct/val_total:.4f}")

# ---------- 5. Save ----------
torch.save(model.state_dict(), 'dog_cat_model.pt')

size_mb = os.path.getsize('dog_cat_model.pt') / 1e6
print(f"Saved model size: {size_mb:.2f} MB")
