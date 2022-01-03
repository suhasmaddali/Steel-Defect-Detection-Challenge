from flask import Flask, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
import os
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)


MODEL_PATH = 'Models/VGG19_weights.h5'

model = load_model(MODEL_PATH)

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    return np.argmax(model.predict(img), axis = 1)

@app.route('/predict', methods = ['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)
    return jsonify(prediction = model_prediction(img_bytes))

@app.route('/', methods = ['GET'])
def index():
    return 'Machine Learning Inference'














if __name__ == '__main__':
    app.run(debug=True)