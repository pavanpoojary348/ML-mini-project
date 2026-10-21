# generate_test_audio.py
"""Generate a short dummy WAV file for testing the genre prediction script.

Creates `data/gtzan/sample.wav` – a 3‑second 440 Hz sine wave.
"""

import os
import numpy as np
import soundfile as sf

# Project root (directory containing this script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_dir = os.path.join(project_root, "data", "gtzan")
os.makedirs(out_dir, exist_ok=True)

sr = 22050
duration = 3.0
freq = 440.0

t = np.linspace(0, duration, int(sr * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * freq * t)

out_path = os.path.join(out_dir, "sample.wav")
sf.write(out_path, audio, sr)

print("Dummy audio file created at:", out_path)
