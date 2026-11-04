"""Flask Web Application for Music Genre Classification & Acoustic Analysis.

Provides an interactive user interface:
1. File upload (.mp3, .wav, .flac, .ogg) and live browser microphone recording.
2. Dual-model ensemble genre classification with whole-song aggregation.
3. Acoustic feature profiling: Estimated BPM, Spectral Centroid, RMS, Dynamic Range.
4. Dynamic 128-band Log-Mel spectrogram visualization.
"""

import os
import sys
import uuid
import tempfile
import io
import base64

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import numpy as np
import torch.nn.functional as F
import soundfile as sf
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gtzan_project.models import EnhancedSECNN, MobileNetV2Classifier
from gtzan_project.dataset import GENRES


# ==============================================================================
# SECTION 1: FLASK APP CONFIGURATION & PRETRAINED MODEL LOADING
# ==============================================================================
# What it does:
#   1. Initializes the Flask server and sets a 50MB max upload limit.
#   2. Instantiates both EnhancedSECNN and MobileNetV2Classifier.
#   3. Loads trained weights (best_secnn_3s.pt and best_mobilenet_3s.pt).
#   4. Sets models to evaluation mode (model.eval()) and maps to CPU/CUDA.
# ==============================================================================

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "results_gtzan_high_acc")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    """Instantiates models, loads trained weights, and sets eval mode."""
    secnn = EnhancedSECNN(num_classes=10, dropout_rate=0.0).to(DEVICE)
    mob   = MobileNetV2Classifier(num_classes=10, dropout_rate=0.0).to(DEVICE)
    for model, name in [(secnn, "best_secnn_3s.pt"), (mob, "best_mobilenet_3s.pt")]:
        p = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE)
            model.load_state_dict(ck.get("model_state_dict", ck))
        model.eval()
    return secnn, mob


print("Loading models...")
SECNN, MOBILENET = load_models()
print(f"Ready on {DEVICE}")


# ==============================================================================
# SECTION 2: ACOUSTIC FEATURE PROFILER (BPM, BRIGHTNESS, ENERGY)
# ==============================================================================
# What it does: Computes physical audio characteristics:
#   1. RMS Energy: Root-mean-square amplitude measuring audio power.
#   2. Dynamic Range: Ratio between peak audio amplitude and RMS level (in dB).
#   3. Spectral Centroid (Brightness): Center of mass of the frequency spectrum;
#      quantifies whether the music sounds dark/bass-heavy or bright/trebly.
#   4. Tempo / BPM: Calculates onset autocorrelation over the short-term energy
#      envelope to estimate beats per minute (BPM).
# ==============================================================================

def extract_acoustic_features(data, sr=22050):
    """Compute key acoustic descriptors: BPM, spectral centroid, RMS energy, dynamic range."""
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    dyn_range = round(float(20 * np.log10((peak + 1e-9) / (rms + 1e-9))), 1)

    # 1. Spectral Centroid Calculation (Brightness in Hz)
    clip_len = min(len(data), sr * 5)
    data_fft = data[:clip_len]
    mags = np.abs(np.fft.rfft(data_fft))
    freqs = np.fft.rfftfreq(len(data_fft), 1.0 / sr)
    centroid = float(np.sum(freqs * mags) / (np.sum(mags) + 1e-9))

    if centroid < 1300:
        desc = "Warm & Bass-heavy (Mellow)"
    elif centroid < 2400:
        desc = "Balanced Midrange (Acoustic)"
    elif centroid < 3800:
        desc = "Bright & Crisp (Treble)"
    else:
        desc = "Sharp & High-Energy (Aggressive)"

    # 2. Tempo / BPM Estimation via Energy Envelope Onset Autocorrelation
    frame_size = int(sr * 0.05)
    hop_size = int(sr * 0.02)
    fps = sr / hop_size
    num_frames = (len(data) - frame_size) // hop_size
    if num_frames > 40:
        energy = np.array([np.sum(data[i*hop_size:i*hop_size+frame_size]**2) for i in range(min(num_frames, 800))])
        onset = np.diff(energy)
        onset[onset < 0] = 0
        onset = onset - np.mean(onset)
        corr = np.correlate(onset, onset, mode='full')
        corr = corr[len(onset)-1:]
        min_lag = int(fps * 60 / 200) # Maximum 200 BPM
        max_lag = int(fps * 60 / 60)  # Minimum 60 BPM
        if max_lag < len(corr):
            lag = min_lag + np.argmax(corr[min_lag:max_lag])
            bpm = round(float(60 * fps / lag))
        else:
            bpm = 120
    else:
        bpm = 120

    return {
        "bpm": int(bpm),
        "brightness_hz": round(centroid, 1),
        "brightness_desc": desc,
        "rms_energy": round(rms, 3),
        "dynamic_range_db": f"{dyn_range} dB"
    }


# ==============================================================================
# SECTION 3: MEL SPECTROGRAM IMAGE RENDERER
# ==============================================================================
# What it does: Takes the 128-band Log-Mel spectrogram evaluated by the model,
# plots it using matplotlib with a dark theme, and converts it to a base64 string
# so the web browser can display it instantly without saving temporary image files.
# ==============================================================================

def generate_spectrogram_image(first_chunk_db):
    """Render 128-band Log-Mel Spectrogram as base64-encoded PNG image."""
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="#080718")
    ax.set_facecolor("#080718")
    im = ax.imshow(first_chunk_db, origin="lower", aspect="auto", cmap="magma")

    ax.set_title("128-Band Log-Mel Spectrogram (AI Input)", color="#c4b5fd", fontsize=10, fontweight="bold", pad=8)
    ax.set_ylabel("Mel Frequency Bands", color="#ffffffb3", fontsize=8)
    ax.set_xlabel("Time Frames (~3.0s slice)", color="#ffffffb3", fontsize=8)
    ax.tick_params(colors="#ffffff99", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#8b5cf666")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("dB Power", color="#ffffffb3", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#ffffff80", labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#ffffff99")
    cbar.outline.set_edgecolor("#8b5cf64d")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ==============================================================================
# SECTION 4: DUAL-MODEL ENSEMBLE INFERENCE ENGINE
# ==============================================================================
# What it does:
#   1. Ingests any audio file, downmixes to mono, and resamples to 22,050 Hz.
#   2. Partitions audio into consecutive 3-second segments.
#   3. For each segment, extracts 128-band Log-Mel spectrogram.
#   4. Runs both Enhanced SE-CNN and MobileNetV2 and applies soft-voting:
#      P_ens = 0.45 * P_secnn + 0.55 * P_mobilenet.
#   5. Averages predictions across all segments for robust whole-song classification.
# ==============================================================================

def run_inference(path):
    """Run dual-model ensemble prediction and extract spectrogram & acoustic features."""
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        # Fallback to torchaudio if soundfile cannot read container
        tensor_data, sr = torchaudio.load(path)
        data = tensor_data.numpy().mean(axis=0)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 22050:
        data = torchaudio.transforms.Resample(sr, 22050)(torch.from_numpy(data)).numpy()

    duration_sec = round(len(data) / 22050.0, 1)
    acoustic_features = extract_acoustic_features(data, 22050)

    mel = torchaudio.transforms.MelSpectrogram(22050, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
    db  = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    cs  = int(22050 * 3.0)
    probs = []
    first_chunk_db = None

    with torch.no_grad():
        num_chunks = max(1, len(data) // cs)
        for i in range(num_chunks):
            chunk = data[i*cs:i*cs+cs]
            if len(chunk) < cs:
                chunk = np.pad(chunk, (0, cs - len(chunk)), mode="wrap")
            mel_raw = db(mel(torch.from_numpy(chunk).unsqueeze(0))).squeeze(0).numpy()
            if first_chunk_db is None:
                first_chunk_db = mel_raw
            s = (mel_raw - 4.88) / 15.84
            t = F.interpolate(
                torch.from_numpy(s).unsqueeze(0).float().unsqueeze(0),
                (224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            
            # Calibrated Soft-Voting Formula
            p = 0.45 * F.softmax(SECNN(t), 1).cpu().numpy()[0] + 0.55 * F.softmax(MOBILENET(t), 1).cpu().numpy()[0]
            probs.append(p)

    # Whole-Song Average Aggregation
    avg = np.mean(probs, axis=0)
    genre_dict = {g: round(float(avg[i]) * 100, 2) for i, g in enumerate(GENRES)}
    spectrogram_b64 = generate_spectrogram_image(first_chunk_db)

    return {
        "genre_probs": genre_dict,
        "acoustic_features": acoustic_features,
        "spectrogram_b64": spectrogram_b64,
        "duration_sec": duration_sec,
        "num_segments": num_chunks
    }


# ==============================================================================
# SECTION 5: FLASK WEB ROUTING & API ENDPOINTS
# ==============================================================================
# Endpoints:
#   - GET /: Renders main interactive Web UI (index.html).
#   - GET /results/<file>: Serves confusion matrices and benchmark charts.
#   - GET /api/metrics: Returns JSON test summary metrics.
#   - POST /predict: Accepts audio upload or live mic recording, runs inference,
#     and returns predicted genre, confidence, features, and spectrogram.
# ==============================================================================

@app.route("/")
def index():
    """Renders the dark-mode web application."""
    return render_template("index.html")


@app.route("/results/<path:filename>")
def results_static(filename):
    """Serves static benchmark assets (confusion matrices, comparison charts)."""
    return send_from_directory(WEIGHTS_DIR, filename)


@app.route("/api/metrics")
def get_metrics():
    """Returns official test set benchmark metrics in JSON format."""
    summary_path = os.path.join(WEIGHTS_DIR, "high_acc_test_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    """Handles audio upload or live microphone input, returning genre probabilities."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a"]:
        return jsonify({"error": f"Unsupported format {ext}"}), 400

    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    try:
        f.save(tmp)
        res = run_inference(tmp)
        sorted_genres = sorted(res["genre_probs"].items(), key=lambda x: x[1], reverse=True)
        return jsonify({
            "top_genre": sorted_genres[0][0],
            "top_prob": sorted_genres[0][1],
            "all_genres": [{"genre": g, "prob": p} for g, p in sorted_genres],
            "acoustic_features": res["acoustic_features"],
            "spectrogram_b64": res["spectrogram_b64"],
            "duration_sec": res["duration_sec"],
            "num_segments": res["num_segments"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ==============================================================================
# SECTION 6: ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
