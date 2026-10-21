import os
import numpy as np
import soundfile as sf

DEMOS_DIR = os.path.join(os.path.dirname(__file__), "static", "demos")
os.makedirs(DEMOS_DIR, exist_ok=True)
sr = 22050
duration = 5.0
total_samples = int(sr * duration)
t = np.linspace(0, duration, total_samples, endpoint=False)

# 1. Classical Piano Demo
def make_classical():
    audio = np.zeros(total_samples)
    notes = [
        (0.0, 261.63), # C4
        (0.8, 329.63), # E4
        (1.6, 392.00), # G4
        (2.4, 493.88), # B4
        (3.2, 523.25), # C5
    ]
    for start_t, freq in notes:
        idx_start = int(start_t * sr)
        note_len = total_samples - idx_start
        t_note = np.linspace(0, note_len / sr, note_len, endpoint=False)
        envelope = np.exp(-1.8 * t_note)
        # Fundamental + rich piano harmonics
        tone = (1.0 * np.sin(2 * np.pi * freq * t_note) +
                0.5 * np.sin(2 * np.pi * 2 * freq * t_note) +
                0.25 * np.sin(2 * np.pi * 3 * freq * t_note) +
                0.1 * np.sin(2 * np.pi * 4 * freq * t_note)) * envelope
        audio[idx_start:] += tone * 0.4
    return np.clip(audio, -0.95, 0.95)

# 2. Metal Drive Demo
def make_metal():
    audio = np.zeros(total_samples)
    bpm = 150
    beat_dur = 60.0 / bpm
    # Distorted power chords (E2 = 82.4Hz, B2 = 123.5Hz)
    chord = np.sin(2 * np.pi * 82.4 * t) + 0.8 * np.sin(2 * np.pi * 123.5 * t)
    distorted = np.tanh(4.0 * chord)
    # Double bass drum kicks + metallic cymbals
    drum = np.zeros(total_samples)
    for b in range(int(duration / (beat_dur / 2))):
        kick_start = int(b * (beat_dur / 2) * sr)
        k_len = min(int(0.12 * sr), total_samples - kick_start)
        k_t = np.linspace(0, k_len / sr, k_len)
        drum[kick_start:kick_start + k_len] += np.sin(2 * np.pi * 60 * np.exp(-15 * k_t) * k_t) * 0.6
        # Cymbal splash (white noise)
        c_len = min(int(0.08 * sr), total_samples - kick_start)
        drum[kick_start:kick_start + c_len] += np.random.randn(c_len) * 0.15
    audio = 0.5 * distorted + 0.5 * drum
    return np.clip(audio, -0.95, 0.95)

# 3. Disco Beat Demo
def make_disco():
    audio = np.zeros(total_samples)
    bpm = 124
    beat_dur = 60.0 / bpm
    # 4-on-the-floor kick
    for b in range(int(duration / beat_dur)):
        k_start = int(b * beat_dur * sr)
        k_len = min(int(0.18 * sr), total_samples - k_start)
        k_t = np.linspace(0, k_len / sr, k_len)
        audio[k_start:k_start + k_len] += np.sin(2 * np.pi * (140 * np.exp(-25 * k_t) + 45) * k_t) * 0.7
        # Offbeat open hi-hat
        hh_start = int((b + 0.5) * beat_dur * sr)
        if hh_start < total_samples:
            hh_len = min(int(0.12 * sr), total_samples - hh_start)
            hh_t = np.linspace(0, hh_len / sr, hh_len)
            noise = np.random.randn(hh_len) * np.exp(-18 * hh_t)
            audio[hh_start:hh_start + hh_len] += noise * 0.35
    # Funky bassline
    bass = np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * (bpm/60) * 2 * t))
    audio += 0.3 * bass
    return np.clip(audio, -0.95, 0.95)

# 4. Blues Riff Demo
def make_blues():
    audio = np.zeros(total_samples)
    # Slow blues shuffle with walking bass notes
    notes = [82.4, 98.0, 110.0, 116.5, 123.5, 116.5, 110.0, 98.0]
    step = 0.55
    for i, freq in enumerate(notes):
        n_start = int(i * step * sr)
        if n_start >= total_samples: break
        n_len = min(int(step * 0.9 * sr), total_samples - n_start)
        n_t = np.linspace(0, n_len / sr, n_len)
        env = np.exp(-2.5 * n_t)
        tone = (np.sin(2 * np.pi * freq * n_t) + 0.4 * np.sin(2 * np.pi * 2 * freq * n_t)) * env
        audio[n_start:n_start + n_len] += tone * 0.6
    return np.clip(audio, -0.95, 0.95)

# 5. Hip-Hop 808 Groove Demo
def make_hiphop():
    audio = np.zeros(total_samples)
    bpm = 85
    beat_dur = 60.0 / bpm
    # Deep 808 kicks on beat 1 and syncopated beat 3.5
    for b in [0, 2.5, 4, 6.5]:
        k_start = int(b * beat_dur * sr)
        if k_start >= total_samples: break
        k_len = min(int(0.45 * sr), total_samples - k_start)
        k_t = np.linspace(0, k_len / sr, k_len)
        freq = 65.0 * np.exp(-4 * k_t) + 38.0
        audio[k_start:k_start + k_len] += np.sin(2 * np.pi * freq * k_t) * 0.85
    # Snare/Clap on beat 2 and 4
    for b in [1, 3, 5]:
        s_start = int(b * beat_dur * sr)
        if s_start >= total_samples: break
        s_len = min(int(0.18 * sr), total_samples - s_start)
        s_t = np.linspace(0, s_len / sr, s_len)
        noise = (np.random.randn(s_len) * 0.5 + np.sin(2 * np.pi * 200 * s_t) * 0.5) * np.exp(-12 * s_t)
        audio[s_start:s_start + s_len] += noise * 0.6
    return np.clip(audio, -0.95, 0.95)

sf.write(os.path.join(DEMOS_DIR, "classical_demo.wav"), make_classical(), sr)
sf.write(os.path.join(DEMOS_DIR, "metal_demo.wav"), make_metal(), sr)
sf.write(os.path.join(DEMOS_DIR, "disco_demo.wav"), make_disco(), sr)
sf.write(os.path.join(DEMOS_DIR, "blues_demo.wav"), make_blues(), sr)
sf.write(os.path.join(DEMOS_DIR, "hiphop_demo.wav"), make_hiphop(), sr)

print("Created 5 audio demo tracks successfully in static/demos/")
