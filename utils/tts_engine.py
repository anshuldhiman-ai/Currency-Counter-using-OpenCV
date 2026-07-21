"""
tts_engine.py

Dual-mode TTS engine:
  1. OFFLINE  – pyttsx3 (robotic but zero-latency, no internet needed)
  2. SARVAM  – Sarvam AI bulbul:v3 API (natural Indian-accented voices,
               requires internet + API key)

Verified working voices only (38 total for bulbul:v3).
Includes gender filtering (Female / Male / All) and keyboard shortcuts.
Speech is queued in a background thread so it never blocks the video loop.
Audio playback for Sarvam uses native Windows MCI API (winmm.dll via ctypes).
"""

import ctypes
import os
import queue
import tempfile
import threading
import time

import pyttsx3

# ---------------------------------------------------------------------------
# Windows MCI helper (native MP3 playback, no third-party libs needed)
# ---------------------------------------------------------------------------
_winmm = ctypes.windll.winmm


def _mci_send(command: str) -> str:
    """Send a command string to the Windows MCI subsystem and return the result."""
    buf = ctypes.create_unicode_buffer(256)
    err = _winmm.mciSendStringW(command, buf, 255, 0)
    if err:
        err_buf = ctypes.create_unicode_buffer(256)
        _winmm.mciGetErrorStringW(err, err_buf, 255)
        raise RuntimeError(f"MCI error {err}: {err_buf.value}")
    return buf.value


# ---------------------------------------------------------------------------
# Sarvam AI configuration
# ---------------------------------------------------------------------------
SARVAM_API_KEY = "sk_y8ni372y_dbrAlcySsSmTIPOeT7NzGDpL"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech/stream"

# VERIFIED WORKING VOICES ONLY for bulbul:v3 model
# Filtered out 9 non-working / incompatible voices (anushka, manisha, vidya, arya, amelia, sophia, abhilash, karun, hitesh)
SARVAM_VOICES = [
    # ── Female (16 verified) ──────────────────────────────────
    ("ritu",     "Ritu",     "Female"),
    ("priya",    "Priya",    "Female"),
    ("neha",     "Neha",     "Female"),
    ("pooja",    "Pooja",    "Female"),
    ("simran",   "Simran",   "Female"),
    ("kavya",    "Kavya",    "Female"),
    ("ishita",   "Ishita",   "Female"),
    ("shreya",   "Shreya",   "Female"),
    ("roopa",    "Roopa",    "Female"),
    ("suhani",   "Suhani",   "Female"),
    ("kavitha",  "Kavitha",  "Female"),
    ("shruti",   "Shruti",   "Female"),
    ("tanya",    "Tanya",    "Female"),
    ("rupali",   "Rupali",   "Female"),
    ("mani",     "Mani",     "Female"),
    ("niharika", "Niharika", "Female"),
    # ── Male (22 verified) ────────────────────────────────────
    ("aditya",   "Aditya",   "Male"),
    ("rahul",    "Rahul",    "Male"),
    ("rohan",    "Rohan",    "Male"),
    ("amit",     "Amit",     "Male"),
    ("dev",      "Dev",      "Male"),
    ("ratan",    "Ratan",    "Male"),
    ("varun",    "Varun",    "Male"),
    ("manan",    "Manan",    "Male"),
    ("sumit",    "Sumit",    "Male"),
    ("kabir",    "Kabir",    "Male"),
    ("aayan",    "Aayan",    "Male"),
    ("shubh",    "Shubh",    "Male"),
    ("ashutosh", "Ashutosh", "Male"),
    ("advait",   "Advait",   "Male"),
    ("anand",    "Anand",    "Male"),
    ("tarun",    "Tarun",    "Male"),
    ("sunny",    "Sunny",    "Male"),
    ("gokul",    "Gokul",    "Male"),
    ("vijay",    "Vijay",    "Male"),
    ("mohit",    "Mohit",    "Male"),
    ("rehan",    "Rehan",    "Male"),
    ("soham",    "Soham",    "Male"),
]

# Quick lookup set
SARVAM_VOICE_KEYS = [v[0] for v in SARVAM_VOICES]

# Engine mode constants
MODE_OFFLINE = "offline"
MODE_SARVAM  = "sarvam"


class TTSEngine:
    """
    Dual-mode text-to-speech engine with non-blocking speech queue and gender filtering.

    Parameters
    ----------
    mode : str
        Initial engine mode: "offline" (pyttsx3) or "sarvam" (Sarvam AI).
    voice : str
        Initial Sarvam voice key (e.g. "priya"). Ignored for offline mode.
    rate : int
        Speech rate for pyttsx3 offline engine.
    volume : float
        Volume for pyttsx3 offline engine.
    """

    def __init__(self, mode=MODE_SARVAM, voice="priya", rate=175, volume=1.0):
        self._mode = mode
        self._gender_filter = "female"  # "female", "male", or "all"
        self._voice_index = 0
        
        # Resolve initial voice
        if voice in SARVAM_VOICE_KEYS:
            self._voice_index = SARVAM_VOICE_KEYS.index(voice)
            self._gender_filter = SARVAM_VOICES[self._voice_index][2].lower()

        self._rate = rate
        self._volume = volume
        self._queue = queue.Queue()
        self._lock = threading.Lock()

        # Counter for unique MCI alias names (avoids collisions)
        self._mci_counter = 0

        # Temp directory for Sarvam audio files
        self._tmp_dir = tempfile.mkdtemp(prefix="inr_tts_")

        # Start background worker
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Internal Voice Filter Helper ──────────────────────────────────────────

    def _get_filtered_indices(self):
        """Return indices of voices matching the active gender filter."""
        if self._gender_filter == "female":
            return [i for i, v in enumerate(SARVAM_VOICES) if v[2].lower() == "female"]
        elif self._gender_filter == "male":
            return [i for i, v in enumerate(SARVAM_VOICES) if v[2].lower() == "male"]
        return list(range(len(SARVAM_VOICES)))

    # ── Public API ─────────────────────────────────────────────────────────────

    def say(self, text: str, replace_pending=True):
        """Non-blocking: queues text to be spoken. Drains stale pending items by default."""
        if replace_pending:
            try:
                with self._queue.mutex:
                    self._queue.queue.clear()
            except Exception:
                pass
        self._queue.put(text)

    def stop(self):
        """Signal the worker to exit."""
        self._queue.put(None)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def gender_filter(self) -> str:
        with self._lock:
            return self._gender_filter

    @property
    def voice_key(self) -> str:
        with self._lock:
            return SARVAM_VOICES[self._voice_index][0]

    @property
    def voice_display(self) -> str:
        with self._lock:
            v = SARVAM_VOICES[self._voice_index]
            return f"{v[1]} ({v[2]})"

    @property
    def engine_display(self) -> str:
        """Human-readable string for the HUD: engine + voice info."""
        with self._lock:
            if self._mode == MODE_OFFLINE:
                return "Offline (pyttsx3)"
            v = SARVAM_VOICES[self._voice_index]
            filter_tag = f"[{self._gender_filter.upper()}] " if self._gender_filter != "all" else ""
            return f"Sarvam AI - {filter_tag}{v[1]} ({v[2]})"

    def toggle_engine(self) -> str:
        """Switch between offline and Sarvam modes. Returns new mode display."""
        with self._lock:
            if self._mode == MODE_OFFLINE:
                self._mode = MODE_SARVAM
            else:
                self._mode = MODE_OFFLINE
            return self._mode

    def toggle_gender(self) -> str:
        """Cycle gender filter: Female -> Male -> All -> Female... Returns active gender string."""
        with self._lock:
            if self._gender_filter == "female":
                self._gender_filter = "male"
            elif self._gender_filter == "male":
                self._gender_filter = "all"
            else:
                self._gender_filter = "female"

            # Set voice index to the first voice matching the new filter
            filtered = self._get_filtered_indices()
            if self._voice_index not in filtered and filtered:
                self._voice_index = filtered[0]
            
            return self._gender_filter

    def next_voice(self) -> str:
        """Cycle to the next voice in the current gender filter."""
        with self._lock:
            filtered = self._get_filtered_indices()
            if not filtered:
                filtered = list(range(len(SARVAM_VOICES)))

            if self._voice_index in filtered:
                curr_pos = filtered.index(self._voice_index)
                next_pos = (curr_pos + 1) % len(filtered)
                self._voice_index = filtered[next_pos]
            else:
                self._voice_index = filtered[0]

            v = SARVAM_VOICES[self._voice_index]
            return f"{v[1]} ({v[2]})"

    def prev_voice(self) -> str:
        """Cycle to the previous voice in the current gender filter."""
        with self._lock:
            filtered = self._get_filtered_indices()
            if not filtered:
                filtered = list(range(len(SARVAM_VOICES)))

            if self._voice_index in filtered:
                curr_pos = filtered.index(self._voice_index)
                prev_pos = (curr_pos - 1) % len(filtered)
                self._voice_index = filtered[prev_pos]
            else:
                self._voice_index = filtered[0]

            v = SARVAM_VOICES[self._voice_index]
            return f"{v[1]} ({v[2]})"

    # ── Background worker ──────────────────────────────────────────────────────

    def _worker(self):
        """Runs in a daemon thread; pops text and speaks using the active engine."""
        offline_engine = pyttsx3.init()
        offline_engine.setProperty("rate", self._rate)
        offline_engine.setProperty("volume", self._volume)

        while True:
            text = self._queue.get()
            if text is None:
                break

            with self._lock:
                current_mode = self._mode
                voice_idx = self._voice_index

            if current_mode == MODE_SARVAM:
                self._speak_sarvam(text, voice_idx)
            else:
                offline_engine.say(text)
                offline_engine.runAndWait()

    # ── Sarvam AI speech ───────────────────────────────────────────────────────

    def _speak_sarvam(self, text: str, voice_idx: int):
        """Call Sarvam API, save the MP3, and play it via Windows MCI."""
        try:
            import requests as _requests
        except ImportError:
            print("[TTS] 'requests' library not installed — falling back to offline.")
            self._speak_offline_fallback(text)
            return

        voice_key = SARVAM_VOICES[voice_idx][0]
        payload = {
            "text":                 text,
            "target_language_code": "en-IN",
            "speaker":              voice_key,
            "model":                "bulbul:v3",
            "pace":                 1.15,
            "speech_sample_rate":   22050,
            "output_audio_codec":   "mp3",
            "enable_preprocessing": False,
        }
        headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type":         "application/json",
        }

        try:
            resp = _requests.post(
                SARVAM_TTS_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=15,
            )
        except Exception as e:
            print(f"[TTS] Sarvam API request failed: {e} — falling back to offline.")
            self._speak_offline_fallback(text)
            return

        if not resp.ok:
            print(f"[TTS] Sarvam API returned HTTP {resp.status_code} — falling back.")
            self._speak_offline_fallback(text)
            return

        ct = resp.headers.get("Content-Type", "")
        if "application/json" in ct:
            print("[TTS] Sarvam returned JSON error — falling back.")
            self._speak_offline_fallback(text)
            return

        filepath = os.path.join(self._tmp_dir, f"tts_{int(time.time()*1000)}.mp3")
        total = 0
        try:
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
        except Exception as e:
            print(f"[TTS] Failed to save audio: {e}")
            self._speak_offline_fallback(text)
            return

        if total == 0:
            print("[TTS] Empty audio from Sarvam — falling back.")
            self._speak_offline_fallback(text)
            return

        self._play_mp3_mci(filepath)

    def _play_mp3_mci(self, filepath: str):
        """Play an MP3 file synchronously using Windows MCI API."""
        self._mci_counter += 1
        alias = f"inrtts{self._mci_counter}"
        safe_path = filepath.replace("/", "\\")

        try:
            _mci_send(f'open "{safe_path}" type mpegvideo alias {alias}')
            _mci_send(f"play {alias}")

            while True:
                status = _mci_send(f"status {alias} mode")
                if status.lower() in ("stopped", ""):
                    break
                time.sleep(0.05)

        except RuntimeError as e:
            print(f"[TTS] MCI playback error: {e}")
        finally:
            try:
                _mci_send(f"close {alias}")
            except RuntimeError:
                pass
            try:
                os.remove(filepath)
            except OSError:
                pass

    def _speak_offline_fallback(self, text: str):
        """Emergency fallback: use pyttsx3 from within worker thread."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Offline fallback error: {e}")


if __name__ == "__main__":
    tts = TTSEngine(mode=MODE_SARVAM, voice="priya")
    print(f"Engine: {tts.engine_display}")
    tts.say("Testing Sarvam AI voice engine.")
    time.sleep(4)
    print(f"Toggle Gender: {tts.toggle_gender()}")
    print(f"Engine: {tts.engine_display}")
    tts.say("Switched to male voice filter.")
    time.sleep(4)
    tts.stop()
