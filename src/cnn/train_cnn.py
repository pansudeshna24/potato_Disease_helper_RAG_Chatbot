import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.optim import Adam
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

DATASET_PATH = "data/image_reference_index"
MODEL_SAVE_PATH = "src/cnn/models/potato_disease_cnn.pth"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# TRANSFORMS
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# ============================================
# DATASET
# ============================================

train_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================
# MODEL
# ============================================

model = efficientnet_b0(weights="DEFAULT")

num_classes = len(train_dataset.classes)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)

# ============================================
# LOSS + OPTIMIZER
# ============================================

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================
# TRAIN LOOP
# ============================================

print("\nStarting EfficientNet Training...")
print(f"Classes: {train_dataset.classes}")
print(f"Device: {DEVICE}\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item(), acc=accuracy)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(
        f"Epoch {epoch+1}: "
        f"Loss={epoch_loss:.4f}, "
        f"Accuracy={epoch_acc:.2f}%"
    )

# ============================================
# SAVE MODEL
# ============================================

os.makedirs("src/cnn/models", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "classes": train_dataset.classes
}

torch.save(checkpoint, MODEL_SAVE_PATH)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")