from flask import Flask, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
import os
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return 'Machine Learning Inference'


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

if __name__ == '__main__':
    app.run(debug=True)