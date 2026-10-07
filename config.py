"""
Configuration for SilentVoice
"""

import os

# Base paths
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data_correct')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Model settings
MODEL_NAME = 'gesture_model_v2.pkl'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# Training settings
SAMPLES_PER_GESTURE = 200
GESTURES = [0, 1, 2, 3, 4, 5]  # 0 = fist, 1-5 = fingers

# Gesture messages
GESTURE_MESSAGES = {
    0: {"message": "Stop / I need space", "short": "Stop", "color": "#3498db", "icon": "✊"},
    1: {"message": "Yes / I agree", "short": "Yes", "color": "#2ecc71", "icon": "☝️"},
    2: {"message": "Help me", "short": "Help", "color": "#e74c3c", "icon": "✌️"},
    3: {"message": "Water please", "short": "Water", "color": "#f1c40f", "icon": "🤟"},
    4: {"message": "Food please", "short": "Food", "color": "#e67e22", "icon": "🖖"},
    5: {"message": "EMERGENCY! Call for help!", "short": "EMERGENCY", "color": "#ff0000", "icon": "🖐️"},
}

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_ID = 0

# Flask settings
SECRET_KEY = 'silentvoice-secret-key-2024'
DEBUG = False
HOST = '0.0.0.0'
PORT = 5000

# TTS settings
TTS_RATE = 150
TTS_VOLUME = 1.0

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)