"""
Collect Diverse Hand Gesture Data
Collects samples with hand rotated, tilted, and at different distances
"""

import cv2
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.hand_tracker import HandTracker
from config import TRAINING_DATA_DIR, SAMPLES_PER_GESTURE, GESTURES, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_ID

# Increase samples for problematic gestures
SAMPLES_PER_GESTURE = 250  # Increase to 250 for better variety

print("\n" + "=" * 60)
print("📸 DIVERSE HAND GESTURE DATA COLLECTION")
print("=" * 60)
print("\n🎮 Controls:")
print("   Press 0-5 to save sample for that gesture")
print("   Press 'r' to reset current gesture's samples")
print("   Press 'q' to quit and save")
print("\n📋 Instructions for BETTER ACCURACY:")
print("   1. Rotate your hand slightly between samples")
print("   2. Move hand closer/farther from camera")
print("   3. Tilt hand left/right")
print("   4. Move hand up/down in frame")
print("   5. Collect AT LEAST 250 samples per gesture")
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

gesture_names = {0: "Fist (Stop)", 1: "One (Yes)", 2: "Two (Help)",
                 3: "Three (Water)", 4: "Four (Food)", 5: "Five (Emergency)"}

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
hand_tracker.start()

current_gesture = 2  # Start with problematic gesture
print(f"\n👉 Starting with gesture {current_gesture}: {gesture_names[current_gesture]}")
print("   Press 0-5 to switch gestures\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    hand_tracker.process_frame(frame)
    frame = hand_tracker.draw_landmarks(frame)
    landmarks = hand_tracker.extract_landmarks()

    # Show progress bar for all gestures
    y_offset = 30
    for g in GESTURES:
        count = len(all_data[g])
        color = (0, 255, 0) if count >= SAMPLES_PER_GESTURE else (0, 100, 255)

        # Highlight current gesture
        if g == current_gesture:
            cv2.rectangle(frame, (5, y_offset - 20), (250, y_offset + 5), (0, 255, 255), 2)

        # Progress bar
        bar_width = int((count / SAMPLES_PER_GESTURE) * 200)
        cv2.rectangle(frame, (200, y_offset - 15), (200 + bar_width, y_offset - 5), color, -1)

        cv2.putText(frame, f"{g}: {gesture_names[g][:12]}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"{count}/{SAMPLES_PER_GESTURE}", (210, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 30

    # Show detected gesture for reference
    if landmarks:
        # Simple geometry check for feedback
        tip_ids = [4, 8, 12, 16, 20]
        finger_count = 0
        for i, tip_id in enumerate(tip_ids):
            if i == 0:  # Thumb
                if landmarks[tip_id * 3] > landmarks[12]:  # Rough check
                    finger_count += 1
            else:
                if landmarks[tip_id * 3 + 1] < landmarks[(tip_id - 2) * 3 + 1]:
                    finger_count += 1
        cv2.putText(frame, f"Detected (approx): {finger_count} fingers", (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

    # Instructions
    cv2.rectangle(frame, (10, frame.shape[0] - 130), (400, frame.shape[0] - 10), (0, 0, 0), -1)
    cv2.putText(frame, "Press 0-5 to select gesture", (20, frame.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Current: {current_gesture} - {gesture_names[current_gesture]}", (20, frame.shape[0] - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Press SPACE to save | r to reset | q to quit", (20, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Collect DIVERSE Data - Press 0-5 to select gesture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Change gesture (0-5 keys)
    if key in [48, 49, 50, 51, 52, 53]:
        current_gesture = key - 48
        print(f"\n🔁 Switched to gesture {current_gesture}: {gesture_names[current_gesture]}")

    # Save sample (SPACE key)
    elif key == 32:  # SPACE
        if landmarks:
            if len(all_data[current_gesture]) < SAMPLES_PER_GESTURE:
                all_data[current_gesture].append(landmarks)
                count = len(all_data[current_gesture])
                print(f"✅ Saved {count}/{SAMPLES_PER_GESTURE} for {gesture_names[current_gesture]}")

                # Show feedback on frame
                cv2.putText(frame, "SAVED!", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"⚠️ Gesture {current_gesture} already has {SAMPLES_PER_GESTURE} samples")
        else:
            print(f"⚠️ No hand detected! Show {gesture_names[current_gesture]} clearly")

    # Reset current gesture's data
    elif key == ord('r'):
        if len(all_data[current_gesture]) > 0:
            confirm = input(
                f"Reset all {len(all_data[current_gesture])} samples for {gesture_names[current_gesture]}? (y/n): ")
            if confirm.lower() == 'y':
                all_data[current_gesture] = []
                print(f"🗑️ Reset {gesture_names[current_gesture]} data")

    # Quit
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