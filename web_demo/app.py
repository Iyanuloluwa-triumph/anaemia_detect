"""
web_demo/app.py

Flask backend for the Anaemia Classifier demo.

Usage:
    pip install flask tensorflow pillow numpy
    python web_demo/app.py

Then open http://localhost:5000 in a browser.
"""
import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

SEVERITY_CLASSES = ["Non-Anemic", "Anemic"]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'phase2_best.keras')

RECOMMENDATIONS = {
    "Non-Anemic": "Haemoglobin appears within normal range. No anaemia detected from the conjunctival image.",
    "Anemic":     "Signs consistent with anaemia detected. Please refer the patient for clinical evaluation and haemoglobin measurement.",
}

# Load once at startup — avoids reloading on every request
print("Loading model ...")
_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model ready.")


def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Replicates the exact training preprocessing from data_pipeline.py:
      1. Open as RGBA
      2. Alpha-composite onto white background
      3. Square-pad shorter dimension (white borders)
      4. Resize to 224 × 224
      5. MobileNetV2 scale: [0, 255] → [-1, 1]
    Returns shape (1, 224, 224, 3) float32.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Composite onto white
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    _, _, _, alpha = img.split()
    background.paste(img, mask=alpha)
    rgb = background.convert("RGB")

    # Pad to square
    w, h = rgb.size
    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top  = (max_side - h) // 2
    square = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    square.paste(rgb, (pad_left, pad_top))

    # Resize and normalise
    resized = square.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    try:
        img_bytes = file.read()
        arr = preprocess(img_bytes)
        probs = _model(arr, training=False).numpy()[0]  # (2,)

        pred_idx   = int(np.argmax(probs))
        pred_class = SEVERITY_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        return jsonify({
            'predicted_class': pred_class,
            'confidence':      round(confidence * 100, 1),
            'recommendation':  RECOMMENDATIONS[pred_class],
            'probabilities': {
                cls: round(float(probs[i]) * 100, 1)
                for i, cls in enumerate(SEVERITY_CLASSES)
            },
        })

    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
