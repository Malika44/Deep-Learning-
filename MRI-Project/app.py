from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os

IMG_SIZE = (128, 128)             # must match training
CLASS_NAMES = {0: "No Tumor", 1: "Tumor"}

app = Flask(__name__)
model = tf.keras.models.load_model("deep_cnn_brain_tumor.h5")

def preprocess(file_bytes):
    """bytes → (1, H, W, 3) float32 0‑1"""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    tensor = preprocess(img_bytes)
    probs  = model.predict(tensor)[0]          # (2,)
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx]) * 100

    return jsonify({
        "class": CLASS_NAMES[idx],
        "confidence": f"{conf:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
