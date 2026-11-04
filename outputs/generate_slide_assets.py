import os
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchaudio
import torch
from matplotlib.patches import FancyBboxPatch

os.makedirs("outputs", exist_ok=True)

# 1. Generate Mel Spectrogram Visual from static/demos/metal_demo.wav
audio_path = "static/demos/metal_demo.wav"
if os.path.exists(audio_path):
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    samples_3s = int(sr * 3.0)
    clip = data[:samples_3s] if len(data) >= samples_3s else np.pad(data, (0, samples_3s - len(data)))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    spec = amp_to_db(mel_transform(torch.from_numpy(clip).unsqueeze(0))).squeeze(0).numpy()
    
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="#1e293b")
    ax.set_facecolor("#0f172a")
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("128-Band Log-Mel Spectrogram (Metal Track - 3s Window)", color="#f8fafc", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Time Frames (Hop Length = 512)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Mel Frequency Bins (0 - 128)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude (dB)", color="#94a3b8", fontsize=8)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=7)
    plt.tight_layout()
    plt.savefig("outputs/sample_mel_spectrogram.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print("Saved outputs/sample_mel_spectrogram.png")

# 2. Generate a clean visual pipeline flowchart
fig, ax = plt.subplots(figsize=(9.2, 2.6), facecolor="#1e293b")
ax.set_facecolor("#1e293b")
ax.axis("off")

boxes = [
    ("Raw Audio\n(.wav / .mp3 / mic)\n22,050 Hz Mono", 0.10, "#0284c7"),
    ("3-Second\nSegmentation\n10 Slices / Song", 0.32, "#2563eb"),
    ("Log-Mel\nSpectrogram\n128 Bins x 130 Frames", 0.54, "#7c3aed"),
    ("Dual Model\nInference\nSE-CNN + MobileNet", 0.76, "#d946ef"),
    ("Calibrated Soft-Vote\n& Song Average\nFinal Genre Prediction", 0.94, "#059669")
]

for i, (text, x, col) in enumerate(boxes):
    box = FancyBboxPatch((x - 0.08, 0.18), 0.155, 0.64,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         facecolor=col, edgecolor="#f8fafc", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x - 0.002, 0.5, text, ha="center", va="center", color="#ffffff", fontsize=8.5, fontweight="bold")
    if i < len(boxes) - 1:
        next_x = boxes[i+1][1]
        ax.annotate("", xy=(next_x - 0.085, 0.5), xytext=(x + 0.08, 0.5),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color="#38bdf8", lw=2.5))

ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/pipeline_flowchart.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print("Saved outputs/pipeline_flowchart.png")
