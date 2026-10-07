"""
SilentVoice - Main Application
GPU-Accelerated Sign Language Recognition
"""

import cv2
import numpy as np
import time
import csv
import os
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request

from src.hand_tracker import HandTracker
from src.voice_engine import VoiceEngine
from src.gesture_predictor_dml import GesturePredictorDML
from config import (
    MODELS_DIR, GESTURE_MESSAGES, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_ID,
    SECRET_KEY, DEBUG, HOST, PORT, TTS_RATE, TTS_VOLUME, LOGS_DIR
)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Initialize components
hand_tracker = HandTracker(max_num_hands=1)
voice_engine = VoiceEngine(rate=TTS_RATE, volume=TTS_VOLUME)

# Load GPU-accelerated model
MODEL_PATH = os.path.join(MODELS_DIR, 'gesture_nn_dml.pth')
predictor = GesturePredictorDML(MODEL_PATH)

# Global state
camera_thread = None
camera_running = False
current_frame = None
frame_lock = threading.Lock()
camera_active = False
is_starting = False
last_spoken_gesture = -1
last_spoken_time = 0
GESTURE_HISTORY = []
LATEST_GESTURE = {"fingers": 0, "message": "Camera Off", "timestamp": "", "active": False}


def log_message(gesture_id, message):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    GESTURE_HISTORY.insert(0, {"timestamp": timestamp, "fingers": gesture_id, "message": message})
    while len(GESTURE_HISTORY) > 50:
        GESTURE_HISTORY.pop()

    try:
        log_file = os.path.join(LOGS_DIR, "messages.csv")
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if os.path.getsize(log_file) == 0:
                writer.writerow(['timestamp', 'fingers', 'message'])
            writer.writerow([timestamp, gesture_id, message])
    except:
        pass


def camera_loop():
    global camera_running, current_frame, camera_active, last_spoken_gesture, last_spoken_time

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ Could not open camera")
        camera_running = False
        camera_active = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    hand_tracker.start()
    predictor.reset_buffer()
    print("✅ Camera loop started")

    while camera_running:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)

        # Process frame
        hand_tracker.process_frame(frame)
        frame = hand_tracker.draw_landmarks(frame)
        landmarks = hand_tracker.extract_landmarks()

        # Predict gesture using GPU model
        gesture_id = predictor.predict_stable(landmarks)

        # Get message
        gesture_info = GESTURE_MESSAGES.get(gesture_id, GESTURE_MESSAGES[0])

        # Update global gesture
        global LATEST_GESTURE
        LATEST_GESTURE = {
            "fingers": gesture_id,
            "message": gesture_info["message"],
            "short": gesture_info["short"],
            "color": gesture_info["color"],
            "icon": gesture_info.get("icon", "✋"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "active": True
        }

        # Speak on change
        current_time = time.time()
        if gesture_id != last_spoken_gesture and current_time - last_spoken_time > 2:
            try:
                voice_engine.speak(gesture_info["message"], async_mode=True)
            except Exception as e:
                print(f"TTS error: {e}")
            last_spoken_gesture = gesture_id
            last_spoken_time = current_time
            log_message(gesture_id, gesture_info["message"])

        # Draw UI
        cv2.rectangle(frame, (10, 10), (280, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture_id}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        color_bgr = tuple(int(gesture_info["color"][i:i + 2], 16) for i in (1, 3, 5))
        cv2.rectangle(frame, (10, 80), (450, 140), color_bgr, -1)
        cv2.putText(frame, gesture_info["message"], (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # GPU indicator
        cv2.putText(frame, "GPU ACTIVE", (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Store frame
        with frame_lock:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            current_frame = buffer.tobytes()

    cap.release()
    hand_tracker.stop()
    camera_active = False
    print("✅ Camera loop ended")


# ============ FLASK ROUTES ============

@app.route('/')
def index():
    return render_template('dashboard.html',
                           gestures=GESTURE_MESSAGES,
                           latest=LATEST_GESTURE)


@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        _, blank_encoded = cv2.imencode('.jpg', blank)
        blank_frame = blank_encoded.tobytes()

        while True:
            if current_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
            time.sleep(0.033)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/start', methods=['POST'])
def start_camera():
    global camera_running, camera_thread, camera_active, is_starting, last_spoken_gesture, last_spoken_time

    if is_starting:
        return jsonify({'success': False, 'message': 'Already starting...'})

    if camera_running:
        return jsonify({'success': False, 'message': 'Camera already running'})

    is_starting = True

    last_spoken_gesture = -1
    last_spoken_time = 0

    camera_running = True
    camera_active = True
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.daemon = True
    camera_thread.start()

    is_starting = False
    return jsonify({'success': True, 'message': 'Camera started'})


@app.route('/api/stop', methods=['POST'])
def stop_camera():
    global camera_running, camera_active, current_frame

    camera_running = False
    camera_active = False
    current_frame = None

    global LATEST_GESTURE
    LATEST_GESTURE = {"fingers": 0, "message": "Camera Off", "short": "Off",
                      "color": "#666666", "timestamp": "", "active": False}

    return jsonify({'success': True, 'message': 'Camera stopped'})


@app.route('/api/gesture')
def get_gesture():
    return jsonify(LATEST_GESTURE)


@app.route('/api/history')
def get_history():
    return jsonify(GESTURE_HISTORY[:20])


@app.route('/api/speak', methods=['POST'])
def speak():
    data = request.json
    message = data.get('message', '')
    if message:
        try:
            voice_engine.speak(message, async_mode=True)
        except Exception as e:
            print(f"TTS error: {e}")
        return jsonify({'success': True})
    return jsonify({'success': False})


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    global GESTURE_HISTORY
    GESTURE_HISTORY = []
    log_file = os.path.join(LOGS_DIR, "messages.csv")
    if os.path.exists(log_file):
        os.remove(log_file)
    return jsonify({'success': True})


@app.route('/api/stats')
def get_stats():
    total = len(GESTURE_HISTORY)
    counts = {}
    for entry in GESTURE_HISTORY:
        counts[entry['fingers']] = counts.get(entry['fingers'], 0) + 1
    most_common = max(counts.items(), key=lambda x: x[1])[0] if counts else 0
    return jsonify({
        'total_gestures': total,
        'most_common_fingers': most_common,
        'gpu_active': predictor.is_dml and predictor.is_loaded
    })


@app.route('/api/status')
def get_status():
    return jsonify({
        'camera_running': camera_running,
        'camera_active': camera_active,
        'predictor_info': predictor.get_info() if predictor else {}
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🖐️ SILENTVOICE - Sign Language Assistant")
    print("=" * 50)
    print(f"\n✅ Model loaded: {predictor.is_loaded}")
    print(f"✅ GPU Acceleration: {predictor.is_dml}")
    if predictor.is_dml and hasattr(predictor, 'device_name'):
        print(f"✅ Device: {predictor.device_name}")
    print(f"✅ Gestures: 0-5 fingers")
    print("\n🎮 Controls:")
    print("   Click START to begin camera")
    print("   Click STOP to end camera")
    print(f"\n🌐 Open: http://{HOST}:{PORT}")
    print("=" * 50 + "\n")

    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)