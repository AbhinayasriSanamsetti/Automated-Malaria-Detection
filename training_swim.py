# ====================================
# 🧠 SWIN TRANSFORMER TRAINING SCRIPT (Single Google Drive)
# ====================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm
import datetime
from google.colab import drive

# ====================================
# 🔗 MOUNT GOOGLE DRIVE (Single Drive)
# ====================================
drive.mount('/content/drive', force_remount=False)  # force_remount=True if needed

# ====================================
# ⚙️ CONFIGURATION
# ====================================
DATA_DIR = "/content/final_swin_dataset/final_swin_dataset"          # Local dataset path (unzipped)
SAVE_DIR = "/content/drive/MyDrive/swin_results"  # Save all results in your Drive
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUM_CLASSES = 6
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ====================================
# 🧩 DATA TRANSFORMS
# ====================================
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ====================================
# 📂 LOAD DATASETS
# ====================================
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=test_tfms)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

class_names = train_ds.classes
print(f"📚 Classes found: {class_names}")

# ====================================
# 🧠 MODEL SETUP
# ====================================
model = models.swin_t(weights="IMAGENET1K_V1")
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ====================================
# ♻️ RESUME TRAINING SUPPORT
# ====================================
checkpoint_path = os.path.join(SAVE_DIR, "checkpoint.pth")
start_epoch = 0
best_val_acc = 0.0
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

if os.path.exists(checkpoint_path):
    print(f"🔁 Found checkpoint at {checkpoint_path}. Resuming training...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    train_accuracies = checkpoint.get("train_accuracies", [])
    val_accuracies = checkpoint.get("val_accuracies", [])
    print(f"✅ Resumed from epoch {start_epoch} (Best Val Acc = {best_val_acc:.4f})")
else:
    print("🚀 No checkpoint found. Starting from scratch.")

# ====================================
# 🏁 TRAINING INITIALIZATION
# ====================================
early_stop_counter = 0
start_time = datetime.datetime.now()
log_file = os.path.join(SAVE_DIR, "training_log.txt")

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Train_Accuracy,Val_Accuracy\n")

# ====================================
# 🔄 TRAINING LOOP
# ====================================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ----- Validation -----
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler.step()

    print(f"📘 Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{train_acc:.4f},{val_acc:.4f}\n")

    # ----- Save checkpoint -----
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }
    torch.save(checkpoint, checkpoint_path)

    # ----- Early Stopping -----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
        break

# ====================================
# 🧪 TESTING PHASE
# ====================================
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
test_acc = report["accuracy"]

with open(log_file, "a") as f:
    f.write("\n\n=== Final Test Results ===\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    for cls in class_names:
        cls_report = report[cls]
        f.write(f"{cls}: Precision={cls_report['precision']:.4f}, "
                f"Recall={cls_report['recall']:.4f}, F1={cls_report['f1-score']:.4f}\n")

print(f"✅ Final Test Accuracy: {test_acc:.4f}")

# ====================================
# 🗂️ SAVE SUMMARY
# ====================================
end_time = datetime.datetime.now()
summary_file = os.path.join(SAVE_DIR, "summary.txt")
with open(summary_file, "w") as f:
    f.write(f"Training started: {start_time}\n")
    f.write(f"Training ended: {end_time}\n")
    f.write(f"Duration: {end_time - start_time}\n")
    f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
    f.write(f"Final Test Accuracy: {test_acc:.4f}\n")

print("\n✅ Training complete with Early Stopping & Resume Support!")
print(f"All results, logs, and model files saved in: {SAVE_DIR}")
print(f"- 🧩 Best Model: {os.path.join(SAVE_DIR, 'best_model.pth')}")
print(f"- 💾 Checkpoint (for resume): {checkpoint_path}")
