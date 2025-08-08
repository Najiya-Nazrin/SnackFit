import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# === Config ===
data_dir = "Datatset"   # Change if needed, relative to where you run script
batch_size = 16
epochs = 5
learning_rate = 0.001

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# === Load dataset ===
print(f"Loading dataset from {data_dir} ...")
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"Total images found: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# === Split dataset ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# === DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# === Model ===
print("Loading pre-trained ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model = model.to(device)

# === Loss and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training loop ===
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(outputs.argmax(1) == labels)

        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            print(f"  Batch {i+1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size

    print(f"Train Loss: {epoch_loss:.4f}  Train Accuracy: {epoch_acc:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(outputs.argmax(1) == labels)

    val_loss /= val_size
    val_acc = val_corrects.double() / val_size
    print(f"Validation Loss: {val_loss:.4f}  Validation Accuracy: {val_acc:.4f}")

print("\nTraining complete!")

# === Save the model ===
torch.save(model.state_dict(), "food_classifier.pth")
print("Model saved as food_classifier.pth")
