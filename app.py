# app.py  —  Emotion Lens Backend
# Serves three endpoints: /predict/text, /predict/audio, /predict/vision
#
# ─────────────────────────────────────────────────────────────────
#  UPDATE v8  (17-Mar-2026)
#  • TEXT  label order fixed from config.json id2label:
#    {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
#  • Each model now has its own label list — they were all trained
#    with different class orderings
#  • VISION: EfficientNet-B3  ✅
#  • AUDIO : Custom CNN .pt   ✅
# ─────────────────────────────────────────────────────────────────

import io, os, time, torch, torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models
from PIL import Image

# ── Optional imports ───────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[WARN] librosa not installed — audio endpoint will fail")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HF_OK = True
except ImportError:
    HF_OK = False
    print("[WARN] transformers not installed — text endpoint will fail")

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── TEXT label order — from text_model/config.json id2label ───
# {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
TEXT_EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

# ── VISION label order — from checkpoint emotion_names key ────
# ['surprise','fear','disgust','happy','sad','angry','neutral']
VISION_EMOTIONS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# ── AUDIO label order — update if yours differs ───────────────
AUDIO_EMOTIONS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# Paths
TEXT_MODEL_FOLDER  = "text_model"
AUDIO_MODEL_FOLDER = "audio_model"
AUDIO_SAMPLE_RATE  = 22050
AUDIO_N_MELS       = 64
AUDIO_MAX_FRAMES   = 128
VISION_MODEL_PATH  = "models/vision_emotion_rafdb_model.pth"
VISION_IMG_SIZE    = 224   # try 224 first; change to 300 if results seem off


# ═══════════════════════════════════════════════════════════════
#  VISION ARCHITECTURE — EfficientNet-B3 (in_features=1536)
# ═══════════════════════════════════════════════════════════════
class EfficientNetEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = models.efficientnet_b3(weights=None)
        self.efficientnet = base
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),           # 0
            nn.Linear(1536, 512),        # 1
            nn.ReLU(),                   # 2
            nn.BatchNorm1d(512),         # 3
            nn.Dropout(p=0.3),           # 4
            nn.Linear(512, 256),         # 5
            nn.ReLU(),                   # 6
            nn.BatchNorm1d(256),         # 7
            nn.Dropout(p=0.2),           # 8
            nn.Linear(256, num_classes), # 9
        )
    def forward(self, x):
        return self.efficientnet(x)


# ═══════════════════════════════════════════════════════════════
#  AUDIO ARCHITECTURE — CNN over mel-spectrogram
# ═══════════════════════════════════════════════════════════════
class AudioEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ═══════════════════════════════════════════════════════════════

def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_FOLDER)
    m = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_FOLDER)
    m.to(DEVICE).eval()
    print(f"[OK] Text  model loaded  -> {TEXT_MODEL_FOLDER}")
    print(f"     Label order         -> {TEXT_EMOTIONS}")
    return tokenizer, m


def load_vision_model():
    checkpoint = torch.load(VISION_MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]
    m = EfficientNetEmotionModel(num_classes=len(VISION_EMOTIONS))
    m.load_state_dict(state_dict, strict=True)
    m.to(DEVICE).eval()
    print(f"[OK] Vision model loaded -> {VISION_MODEL_PATH}")
    print(f"     Label order         -> {VISION_EMOTIONS}")
    return m


def load_audio_model():
    candidates = [
        "best_model (1).pt", "best_model.pt",
        "full_model (1).pt", "full_model.pt",
        "model_weights (1).pt", "model_weights.pt",
    ]
    weights_path = None
    for name in candidates:
        path = os.path.join(AUDIO_MODEL_FOLDER, name)
        if os.path.exists(path):
            weights_path = path
            break
    if weights_path is None:
        raise FileNotFoundError(f"No .pt file found in '{AUDIO_MODEL_FOLDER}'.")

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        state_dict = (checkpoint.get("model_state_dict")
                      or checkpoint.get("state_dict")
                      or checkpoint.get("model")
                      or checkpoint)
    else:
        state_dict = checkpoint

    m = AudioEmotionCNN(num_classes=len(AUDIO_EMOTIONS))
    m.load_state_dict(state_dict, strict=False)
    m.to(DEVICE).eval()
    print(f"[OK] Audio model loaded  -> {weights_path}")
    print(f"     Label order         -> {AUDIO_EMOTIONS}")
    return m


# ═══════════════════════════════════════════════════════════════
#  LOAD ALL MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════
print(f"\n[INFO] Device: {DEVICE}\n")

text_tokenizer, text_model = load_text_model()  if HF_OK     else (None, None)
vision_model               = load_vision_model()
audio_model                = load_audio_model() if LIBROSA_OK else None


# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════

vision_transform = transforms.Compose([
    transforms.Resize((VISION_IMG_SIZE, VISION_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_text(text):
    enc = text_tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return vision_transform(img).unsqueeze(0).to(DEVICE)

def preprocess_audio(file_bytes):
    y, sr  = librosa.load(io.BytesIO(file_bytes), sr=AUDIO_SAMPLE_RATE, mono=True)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=AUDIO_N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < AUDIO_MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, AUDIO_MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :AUDIO_MAX_FRAMES]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ═══════════════════════════════════════════════════════════════
#  PREDICTION HELPERS — one per model (each has own label list)
# ═══════════════════════════════════════════════════════════════

def make_text_response(logits):
    probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return {
        "emotion":    TEXT_EMOTIONS[top_idx].capitalize(),
        "confidence": float(probs[top_idx]),
        "emotions":   {TEXT_EMOTIONS[i].capitalize(): float(probs[i])
                       for i in range(len(TEXT_EMOTIONS))}
    }

def make_vision_response(logits):
    probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return {
        "emotion":    VISION_EMOTIONS[top_idx].capitalize(),
        "confidence": float(probs[top_idx]),
        "emotions":   {VISION_EMOTIONS[i].capitalize(): float(probs[i])
                       for i in range(len(VISION_EMOTIONS))}
    }

def make_audio_response(logits):
    probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return {
        "emotion":    AUDIO_EMOTIONS[top_idx].capitalize(),
        "confidence": float(probs[top_idx]),
        "emotions":   {AUDIO_EMOTIONS[i].capitalize(): float(probs[i])
                       for i in range(len(AUDIO_EMOTIONS))}
    }


# ═══════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Emotion Lens API v8 — Endpoints: /predict/text  /predict/audio  /predict/vision"

@app.route("/predict/text", methods=["POST"])
def predict_text():
    if text_model is None:
        return jsonify({"error": "Text model not loaded. Run: pip install transformers safetensors"}), 503
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": 'Send JSON body: { "text": "your sentence" }'}), 400
    t0 = time.time()
    with torch.no_grad():
        inputs = preprocess_text(data["text"])
        output = text_model(**inputs)
        result = make_text_response(output.logits)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(result)

@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    if audio_model is None:
        return jsonify({"error": "Audio model not loaded. Run: pip install librosa"}), 503
    if "file" not in request.files:
        return jsonify({"error": "Send a multipart file with key 'file'"}), 400
    t0 = time.time()
    with torch.no_grad():
        tensor = preprocess_audio(request.files["file"].read())
        result = make_audio_response(audio_model(tensor))
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(result)

@app.route("/predict/vision", methods=["POST"])
def predict_vision():
    if "file" not in request.files:
        return jsonify({"error": "Send a multipart file with key 'file'"}), 400
    t0 = time.time()
    with torch.no_grad():
        tensor = preprocess_image(request.files["file"].read())
        result = make_vision_response(vision_model(tensor))
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n[INFO] Starting Emotion Lens API on http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
