"""
Gesture Predictor with DirectML GPU Support
Uses AMD Radeon GPU for fast inference
"""

import torch
import torch.nn as nn
import numpy as np
import os
from collections import Counter


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


class GesturePredictorDML:
    def __init__(self, model_path, buffer_size=5):
        self.model_path = model_path
        self.buffer_size = buffer_size
        self.is_loaded = False
        self.predictions_buffer = []

        # Setup DirectML device
        try:
            import torch_directml
            self.device = torch_directml.device()
            self.device_name = torch_directml.device_name(0)
            self.is_dml = True
            print(f"✅ Using DirectML on: {self.device_name}")
        except ImportError:
            self.device = torch.device('cpu')
            self.is_dml = False
            print(f"⚠️ DirectML not available, using CPU")

        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = GestureNet().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.is_loaded = True
                print(f"✅ Loaded GPU model from {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                self.is_loaded = False
        else:
            print(f"⚠️ Model not found: {self.model_path}")
            print(f"   Run: python training/train_nn_dml.py first")
            self.is_loaded = False

    def predict(self, landmarks):
        """Predict gesture from landmarks (63 features)"""
        if not self.is_loaded or landmarks is None:
            return 0

        # Convert to tensor and move to GPU
        input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return prediction

    def predict_stable(self, landmarks):
        """Predict with stability buffer (prevents flickering)"""
        if not self.is_loaded or landmarks is None:
            return 0

        prediction = self.predict(landmarks)

        self.predictions_buffer.append(prediction)
        if len(self.predictions_buffer) > self.buffer_size:
            self.predictions_buffer.pop(0)

        # Return the most common prediction in buffer
        return Counter(self.predictions_buffer).most_common(1)[0][0]

    def reset_buffer(self):
        """Reset prediction buffer"""
        self.predictions_buffer = []

    def get_info(self):
        """Get predictor info"""
        return {
            'model_loaded': self.is_loaded,
            'device': str(self.device),
            'device_name': self.device_name if hasattr(self, 'device_name') else 'Unknown',
            'buffer_size': self.buffer_size,
            'dml_enabled': self.is_dml
        }