from flask import Flask, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
import os
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')



@app.route('/predict', methods = ['GET', 'POST'])
def infer_image():
    return 'This is just generating a random image'

if __name__ == '__main__':
    app.run(debug=True)