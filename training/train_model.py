"""
Train Random Forest Classifier on Hand Landmark Data
"""

import json
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TRAINING_DATA_DIR, MODELS_DIR, GESTURES, SAMPLES_PER_GESTURE

MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_model_v2.pkl')

print("\n" + "=" * 60)
print("🤖 TRAINING GESTURE CLASSIFIER")
print("=" * 60)

X = []
y = []
gesture_names = {0: "Fist", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

for gesture_id in GESTURES:
    json_file = os.path.join(TRAINING_DATA_DIR, f"gesture_{gesture_id}.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        for sample in data:
            X.append(sample)
            y.append(gesture_id)

        print(f"✅ Loaded {len(data)} samples for gesture {gesture_id} ({gesture_names[gesture_id]})")
    else:
        print(f"⚠️ No data for gesture {gesture_id}")

X = np.array(X)
y = np.array(y)

print(f"\n📊 Total samples: {len(X)}")
print(f"📊 Feature size: {X.shape[1]} (21 landmarks × 3 coordinates)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📊 Training set: {len(X_train)} samples")
print(f"📊 Testing set: {len(X_test)} samples")

# Train
print("\n🔧 Training Random Forest Classifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print("        Predicted →")
print("        0   1   2   3   4   5")
print("       ----------------------")
cm = confusion_matrix(y_test, y_pred)
for i, label in enumerate(GESTURES):
    print(f"True {label} | {cm[i][0]:3d} {cm[i][1]:3d} {cm[i][2]:3d} {cm[i][3]:3d} {cm[i][4]:3d} {cm[i][5]:3d}")

# Per-gesture accuracy
print("\n🎯 Per-gesture accuracy:")
accuracy_list = []
for gesture in GESTURES:
    mask = y_test == gesture
    if np.sum(mask) > 0:
        acc = np.mean(y_pred[mask] == gesture)
        print(f"   Gesture {gesture} ({gesture_names[gesture]}): {acc:.2%}")
        accuracy_list.append(acc)

# Overall accuracy
accuracy = np.mean(y_pred == y_test)
print(f"\n🏆 Overall Accuracy: {accuracy:.2%}")

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")

# Also save as latest for easy loading
latest_path = os.path.join(MODELS_DIR, 'gesture_model_latest.pkl')
joblib.dump(clf, latest_path)

print("\n✅ Training complete!")
print("👉 Run: python app.py")