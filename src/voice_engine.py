"""
Text-to-Speech Engine
"""

import pyttsx3
import threading


class VoiceEngine:
    def __init__(self, rate=150, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Try to use female voice
        voices = self.engine.getProperty('voices')
        if voices:
            for voice in voices:
                if 'female' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

        self.is_speaking = False
        self.last_message = ""

    def speak(self, message, async_mode=True):
        if not message or message == self.last_message:
            return

        self.last_message = message
        self.is_speaking = True

        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(message,))
            thread.daemon = True
            thread.start()
        else:
            self._speak_sync(message)

    def _speak_sync(self, message):
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False

    def stop(self):
        try:
            self.engine.stop()
        except:
            pass
        self.is_speaking = False