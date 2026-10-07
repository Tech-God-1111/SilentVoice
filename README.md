# 🖐️ SilentVoice - Sign Language Communication Assistant

A real-time hand gesture recognition system that converts sign language (finger counting) into text and speech. Designed to help mute, deaf, and speech-impaired individuals communicate effectively.

## 🌟 Features

- **Real-time Hand Tracking** - Detects hand and counts fingers instantly
- **Text-to-Speech** - Spoken output of detected gestures
- **Web Interface** - Accessible from any device on the network
- **Message History** - Logs all communications
- **Emergency Alert** - Special handling for emergency signals
- **Color-Coded Feedback** - Visual indication of message types

## 🎯 Gesture Mapping

| Fingers | Message | Color |
|---------|---------|-------|
| 0 | Stop / I need space | Blue |
| 1 | Yes / I agree | Green |
| 2 | Help me | Red |
| 3 | Water please | Yellow |
| 4 | Food please | Orange |
| 5 | Emergency! Call for help! | Flashing Red |

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- Webcam

### Setup

```bash
# Clone or create project folder
mkdir SilentVoice
cd SilentVoice

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install flask opencv-python mediapipe numpy pyttsx3

# Run the application
python app.py