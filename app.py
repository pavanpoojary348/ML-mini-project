"""Flask web server for Music Genre Classification.

Run with:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os, sys, uuid, tempfile
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, render_template
import torch, numpy as np
import torch.nn.functional as F
from gtzan_project.models import EnhancedSECNN, MobileNetV2Classifier
from gtzan_project.dataset import GENRES

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "results_gtzan_high_acc")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    secnn = EnhancedSECNN(num_classes=10, dropout_rate=0.0).to(DEVICE)
    mob   = MobileNetV2Classifier(num_classes=10, dropout_rate=0.0).to(DEVICE)
    for model, name in [(secnn,"best_secnn_3s.pt"),(mob,"best_mobilenet_3s.pt")]:
        p = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=DEVICE)
            model.load_state_dict(ck.get("model_state_dict", ck))
        model.eval()
    return secnn, mob

print("Loading models...")
SECNN, MOBILENET = load_models()
print(f"Ready on {DEVICE}")

def run_inference(path):
    import soundfile as sf, torchaudio
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1: data = data.mean(axis=1)
    if sr != 22050:
        data = torchaudio.transforms.Resample(sr, 22050)(torch.from_numpy(data)).numpy()
    mel = torchaudio.transforms.MelSpectrogram(22050, n_fft=2048, hop_length=512, n_mels=128, power=2.0)
    db  = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    cs  = int(22050 * 3.0)
    probs = []
    with torch.no_grad():
        for i in range(max(1, len(data)//cs)):
            chunk = data[i*cs:i*cs+cs]
            if len(chunk) < cs: chunk = np.pad(chunk,(0,cs-len(chunk)),mode="wrap")
            s = (db(mel(torch.from_numpy(chunk).unsqueeze(0))).squeeze(0).numpy()-4.88)/15.84
            t = F.interpolate(torch.from_numpy(s).unsqueeze(0).float().unsqueeze(0),(224,224),mode="bilinear",align_corners=False).squeeze(0).repeat(3,1,1).unsqueeze(0).to(DEVICE)
            p = 0.45*F.softmax(SECNN(t),1).cpu().numpy()[0] + 0.55*F.softmax(MOBILENET(t),1).cpu().numpy()[0]
            probs.append(p)
    avg = np.mean(probs,axis=0)
    return {g: round(float(avg[i])*100,2) for i,g in enumerate(GENRES)}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return jsonify({"error":"No file uploaded"}),400
    f = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".wav",".mp3",".ogg",".flac"]: return jsonify({"error":f"Unsupported format {ext}"}),400
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
    try:
        f.save(tmp)
        res = run_inference(tmp)
        s = sorted(res.items(), key=lambda x:x[1], reverse=True)
        return jsonify({"top_genre":s[0][0],"top_prob":s[0][1],"all_genres":[{"genre":g,"prob":p} for g,p in s]})
    except Exception as e: return jsonify({"error":str(e)}),500
    finally:
        if os.path.exists(tmp): os.remove(tmp)

if __name__ == "__main__": app.run(debug=False, port=5000)
