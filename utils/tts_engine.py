"""
tts_engine.py

Thin wrapper around pyttsx3 (fully offline TTS — important for accessibility,
no internet dependency) with a background queue so speech doesn't block the
video loop.
"""

import queue
import threading
import pyttsx3


class TTSEngine:
    def __init__(self, rate=175, volume=1.0):
        self._queue = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        # engine must be created inside the thread it runs on
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        while True:
            text = self._queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, text: str):
        """Non-blocking: queues text to be spoken."""
        self._queue.put(text)

    def stop(self):
        self._queue.put(None)


if __name__ == "__main__":
    import time
    tts = TTSEngine()
    tts.say("Detected 500 rupee note. Total: 2500 rupees.")
    time.sleep(3)
