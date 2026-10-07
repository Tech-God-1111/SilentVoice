"""
Data Augmentation for Hand Gestures
Creates synthetic variations of existing samples
Fixes confusion between 2,3,4 fingers
"""

import json
import numpy as np
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TRAINING_DATA_DIR

# Configuration
AUGMENTATION_MULTIPLIER = 3  # Create 3 synthetic samples per real sample
TARGET_PER_GESTURE = 500  # Target samples per gesture after augmentation

NOISE_LEVEL = 0.02  # Small random noise to landmarks
ROTATION_RANGE = 0.05  # Simulate hand rotation
SCALE_RANGE = 0.03  # Simulate distance changes

print("\n" + "=" * 60)
print("🔄 DATA AUGMENTATION FOR HAND GESTURES")
print("=" * 60)


def augment_sample(landmarks, method='all'):
    """
    Create augmented version of landmarks
    landmarks: list of 63 values (21 x,y,z)
    """
    arr = np.array(landmarks).reshape(21, 3)
    augmented = []

    # Method 1: Add random noise
    if method in ['noise', 'all']:
        noise = np.random.normal(0, NOISE_LEVEL, arr.shape)
        noise_sample = arr + noise
        augmented.append(noise_sample.flatten().tolist())

    # Method 2: Simulate rotation (add bias to x,y)
    if method in ['rotation', 'all']:
        rotation = np.zeros(arr.shape)
        rotation[:, 0] += np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)  # x shift
        rotation[:, 1] += np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)  # y shift
        rotation_sample = arr + rotation
        augmented.append(rotation_sample.flatten().tolist())

    # Method 3: Simulate distance change (scale all landmarks)
    if method in ['scale', 'all']:
        scale = np.random.uniform(1 - SCALE_RANGE, 1 + SCALE_RANGE)
        scale_sample = arr * scale
        augmented.append(scale_sample.flatten().tolist())

    # Method 4: Combination of all
    if method == 'all':
        combined = arr + np.random.normal(0, NOISE_LEVEL, arr.shape)
        combined = combined + np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        combined = combined * np.random.uniform(1 - SCALE_RANGE, 1 + SCALE_RANGE)
        augmented.append(combined.flatten().tolist())

    return augmented


# Process each gesture
for gesture_id in range(0, 6):
    json_file = os.path.join(TRAINING_DATA_DIR, f"gesture_{gesture_id}.json")

    if not os.path.exists(json_file):
        print(f"⚠️ No data for gesture {gesture_id}")
        continue

    with open(json_file, 'r') as f:
        original_data = json.load(f)

    original_count = len(original_data)
    print(f"\n📂 Gesture {gesture_id}: {original_count} original samples")

    # Create augmented samples
    augmented_data = original_data.copy()

    # Calculate how many more we need
    needed = TARGET_PER_GESTURE - original_count
    if needed > 0:
        # Generate required samples
        while len(augmented_data) < TARGET_PER_GESTURE:
            # Pick random original sample
            original = random.choice(original_data)
            # Create augmented version
            new_samples = augment_sample(original, method='all')
            augmented_data.extend(new_samples)

        # Trim to exact target
        augmented_data = augmented_data[:TARGET_PER_GESTURE]

        # Save augmented data
        with open(json_file, 'w') as f:
            json.dump(augmented_data, f)

        print(f"   ✅ Augmented to {len(augmented_data)} samples (+{len(augmented_data) - original_count})")
    else:
        print(f"   ✅ Already has {original_count} samples (target: {TARGET_PER_GESTURE})")

print("\n" + "=" * 60)
print("✅ DATA AUGMENTATION COMPLETE!")
print("=" * 60)
print("\n👉 Now run: python training/train_model.py")