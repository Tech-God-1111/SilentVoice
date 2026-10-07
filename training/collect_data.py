"""
Collect Hand Gesture Data for Training
Press 0-5 to save samples for each gesture
"""

import cv2
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.hand_tracker import HandTracker
from config import TRAINING_DATA_DIR, SAMPLES_PER_GESTURE, GESTURES, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_ID

print("\n" + "=" * 60)
print("📸 HAND GESTURE DATA COLLECTION")
print("=" * 60)
print("\n🎮 Controls:")
print("   Press 0-5 to save sample for that gesture")
print("   Press 'q' to quit and save")
print("\n📋 Gesture Mapping:")
print("   0 = Fist (Stop)")
print("   1 = 1 finger (Yes)")
print("   2 = 2 fingers (Help)")
print("   3 = 3 fingers (Water)")
print("   4 = 4 fingers (Food)")
print("   5 = 5 fingers (Emergency)")
print(f"\n🎯 Target: {SAMPLES_PER_GESTURE} samples per gesture")
print("\n" + "=" * 60)

# Initialize
hand_tracker = HandTracker(max_num_hands=1)

# Load existing data
all_data = {g: [] for g in GESTURES}
for gesture_id in GESTURES:
    filename = os.path.join(TRAINING_DATA_DIR, f"gesture_{gesture_id}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_data[gesture_id] = json.load(f)
        print(f"Loaded {len(all_data[gesture_id])} existing samples for gesture {gesture_id}")

# Camera
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
hand_tracker.start()

gesture_names = {0: "Fist", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    hand_tracker.process_frame(frame)
    frame = hand_tracker.draw_landmarks(frame)
    landmarks = hand_tracker.extract_landmarks()

    # Show progress
    y_offset = 30
    for g in GESTURES:
        count = len(all_data[g])
        color = (0, 255, 0) if count >= SAMPLES_PER_GESTURE else (0, 100, 255)
        cv2.putText(frame, f"Gesture {g} ({gesture_names[g]}): {count}/{SAMPLES_PER_GESTURE}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30

    # Instructions
    cv2.rectangle(frame, (10, frame.shape[0] - 80), (350, frame.shape[0] - 10), (0, 0, 0), -1)
    cv2.putText(frame, "Press 0-5 to save that gesture", (20, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Collect Data - Press 0-5 to save", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save sample for gesture 0-5
    if key in [48, 49, 50, 51, 52, 53]:  # Keys 0-5
        gesture_id = key - 48  # 0,1,2,3,4,5
        if landmarks:
            if len(all_data[gesture_id]) < SAMPLES_PER_GESTURE:
                all_data[gesture_id].append(landmarks)
                print(
                    f"✅ Saved sample {len(all_data[gesture_id])}/{SAMPLES_PER_GESTURE} for {gesture_names[gesture_id]}")
            else:
                print(f"⚠️ Gesture {gesture_id} already has {SAMPLES_PER_GESTURE} samples")
        else:
            print(f"⚠️ No hand detected! Show {gesture_names[gesture_id]} clearly")

    elif key == ord('q'):
        break

# Save all data
print("\n💾 Saving training data...")
for gesture_id in GESTURES:
    filename = os.path.join(TRAINING_DATA_DIR, f"gesture_{gesture_id}.json")
    with open(filename, 'w') as f:
        json.dump(all_data[gesture_id], f)
    print(f"   Gesture {gesture_id} ({gesture_names[gesture_id]}): {len(all_data[gesture_id])} samples saved")

cap.release()
cv2.destroyAllWindows()
hand_tracker.release()

print("\n✅ Data collection complete!")
print("👉 Run: python training/train_model.py")