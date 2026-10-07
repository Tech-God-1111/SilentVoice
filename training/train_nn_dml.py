"""
Neural Network Training with DirectML (AMD GPU)
Trains on your 1200 hand gesture samples using GPU acceleration
"""

import json
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TRAINING_DATA_DIR, MODELS_DIR

# Setup DirectML device
print("\n" + "=" * 60)
print("🤖 NEURAL NETWORK TRAINING WITH AMD GPU")
print("=" * 60)

try:
    import torch_directml

    device = torch_directml.device()
    device_name = torch_directml.device_name(0)
    is_dml = True
    print(f"\n✅ AMD GPU DETECTED!")
    print(f"   Device: {device}")
    print(f"   Name: {device_name}")
    print(f"   Status: READY FOR TRAINING")
except ImportError:
    device = torch.device('cpu')
    is_dml = False
    print(f"\n⚠️ DirectML not found, falling back to CPU")

print(f"\n📊 Training Device: {device}")
print(f"{'=' * 60}\n")


# Neural Network Architecture
class GestureNet(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=6):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


# Load data
print("📂 Loading training data...")
X = []
y = []
gesture_names = {0: "Fist (Stop)", 1: "One (Yes)", 2: "Two (Help)",
                 3: "Three (Water)", 4: "Four (Food)", 5: "Five (Emergency)"}

for gesture_id in range(6):
    json_file = os.path.join(TRAINING_DATA_DIR, f"gesture_{gesture_id}.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        for sample in data:
            X.append(sample)
            y.append(gesture_id)
        print(f"   Gesture {gesture_id} ({gesture_names[gesture_id]}): {len(data)} samples")
    else:
        print(f"   ⚠️ Gesture {gesture_id} not found")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"\n📊 Total samples: {len(X)}")
print(f"📊 Feature size: {X.shape[1]} (21 landmarks × 3 coordinates)")
print(f"📊 Classes: 0-5 (6 gestures)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Training set: {len(X_train)} samples")
print(f"📊 Testing set: {len(X_test)} samples")

# Convert to PyTorch tensors and move to device
X_train = torch.tensor(X_train).to(device)
y_train = torch.tensor(y_train).to(device)
X_test = torch.tensor(X_test).to(device)
y_test = torch.tensor(y_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, optimizer
model = GestureNet(input_size=63, hidden_size=128, num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

print(f"\n🔧 Neural Network Architecture:")
print(f"   Input layer: 63 features")
print(f"   Hidden layer 1: 128 neurons + BatchNorm + Dropout")
print(f"   Hidden layer 2: 128 neurons + BatchNorm + Dropout")
print(f"   Hidden layer 3: 64 neurons + BatchNorm")
print(f"   Output layer: 6 classes (gestures 0-5)")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
print(f"\n🚀 Starting training on {device}...")
print(f"{'=' * 60}\n")

num_epochs = 100
best_accuracy = 0
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        test_accuracy = 100 * (predicted == y_test).sum().item() / len(y_test)

    scheduler.step(avg_loss)

    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        model_path = os.path.join(MODELS_DIR, 'gesture_nn_dml.pth')
        torch.save(model.state_dict(), model_path)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}%")

training_time = time.time() - start_time

print(f"\n{'=' * 60}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'=' * 60}")
print(f"⏱️  Training time: {training_time:.2f} seconds")
print(f"🏆 Best test accuracy: {best_accuracy:.2f}%")

# Final evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, y_pred = torch.max(outputs.data, 1)

    # Move to CPU for sklearn metrics
    y_test_cpu = y_test.cpu().numpy()
    y_pred_cpu = y_pred.cpu().numpy()

    print(f"\n📈 CLASSIFICATION REPORT:")
    print(classification_report(y_test_cpu, y_pred_cpu, target_names=[gesture_names[i] for i in range(6)]))

    print(f"\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test_cpu, y_pred_cpu)
    print("        Predicted →")
    print("        0   1   2   3   4   5")
    print("       ----------------------")
    for i in range(6):
        print(f"True {i} | {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d} {cm[i][3]:3d} {cm[i][4]:3d} {cm[i][5]:3d}")

# Save final model (overwrite with best)
final_model_path = os.path.join(MODELS_DIR, 'gesture_nn_dml.pth')
torch.save(model.state_dict(), final_model_path)
print(f"\n💾 Model saved to: {final_model_path}")

# Save a CPU-compatible version for fallback
cpu_model_path = os.path.join(MODELS_DIR, 'gesture_nn_dml_cpu.pth')
cpu_model = GestureNet().to('cpu')
cpu_model.load_state_dict(torch.load(final_model_path, map_location='cpu'))
torch.save(cpu_model.state_dict(), cpu_model_path)
print(f"💾 CPU fallback model saved to: {cpu_model_path}")

# Performance comparison
print(f"\n{'=' * 60}")
print(f"🚀 PERFORMANCE SUMMARY")
print(f"{'=' * 60}")
print(f"Training Device: {device}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Model Size: {os.path.getsize(final_model_path) / 1024 / 1024:.2f} MB")
print(f"Test Accuracy: {best_accuracy:.2f}%")
print(f"\n✅ Model is ready for real-time inference!")
print(f"👉 Run: python app.py")
print(f"{'=' * 60}\n")