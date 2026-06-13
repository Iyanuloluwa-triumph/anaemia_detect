import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

SEVERITY_CLASSES = ["Non-Anemic", "Anemic"]
IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, '..', 'models', 'anaemia_classifier.tflite')

RECOMMENDATIONS = {
    "Non-Anemic": "Haemoglobin appears within normal range. No anaemia detected from the conjunctival image.",
    "Anemic":     "Signs consistent with anaemia detected. Please refer the patient for clinical evaluation and haemoglobin measurement.",
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

print("Loading model ...")
_interpreter = Interpreter(model_path=TFLITE_PATH)
_interpreter.allocate_tensors()
_in_idx  = _interpreter.get_input_details()[0]['index']
_out_idx = _interpreter.get_output_details()[0]['index']
print("Model ready.")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    _, _, _, alpha = img.split()
    background.paste(img, mask=alpha)
    rgb = background.convert("RGB")

    w, h = rgb.size
    max_side = max(w, h)
    square = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    square.paste(rgb, ((max_side - w) // 2, (max_side - h) // 2))

    resized = square.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 127.5 - 1.0
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

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    try:
        arr = preprocess(file.read())
        _interpreter.set_tensor(_in_idx, arr)
        _interpreter.invoke()
        probs = _interpreter.get_tensor(_out_idx)[0]

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
