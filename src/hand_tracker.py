"""
Hand Tracking Module - MediaPipe 0.10+ Compatible
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import time


class HandTracker:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.is_running = False
        self.model_path = self._get_model_path()

        # Initialize detector
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.timestamp_ms = 0

        # Hand connections for drawing
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        print(f"✅ HandTracker initialized")

    def _get_model_path(self):
        local_model = Path("hand_landmarker.task")
        if local_model.exists():
            return str(local_model)

        try:
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, "hand_landmarker.task")
            print("✅ Model downloaded")
            return "hand_landmarker.task"
        except Exception as e:
            raise FileNotFoundError(f"Model not found: {e}")

    def start(self):
        self.is_running = True
        self.timestamp_ms = 0
        print("✅ HandTracker started")

    def stop(self):
        self.is_running = False
        print("✅ HandTracker stopped")

    def process_frame(self, frame):
        if not self.is_running:
            return None

        self.timestamp_ms += 33
        if self.timestamp_ms > 300000:
            self.timestamp_ms = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            self.results = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        except Exception as e:
            self.results = None

        return self.results

    def extract_landmarks(self):
        """Extract 63 features (21 landmarks × 3 coordinates)"""
        if not self.results or not self.results.hand_landmarks:
            return None

        landmarks = []
        for lm in self.results.hand_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks

    def draw_landmarks(self, frame):
        if not self.is_running or not self.results or not self.results.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        for hand_landmarks in self.results.hand_landmarks:
            # Draw connections
            for connection in self.hand_connections:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Draw landmarks
            tip_ids = [4, 8, 12, 16, 20]
            for i, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if i in tip_ids:
                    color = (0, 0, 255)
                    radius = 6
                else:
                    color = (255, 0, 0)
                    radius = 3

                cv2.circle(frame, (x, y), radius, color, -1)

        return frame

    def release(self):
        if self.detector:
            try:
                self.detector.close()
            except:
                pass
        self.is_running = False
        print("✅ HandTracker released")