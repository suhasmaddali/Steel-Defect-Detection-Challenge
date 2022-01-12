from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
MODEL_PATH = 'Models/VGG19_weights.h5'

model = load_model(MODEL_PATH)
# model.summary()
app = Flask(__name__)
TRAIN_IMAGE_PATH = '../Steel Defect Detection Dataset/train_images/00f6e702c.jpg'

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis = 0)
    preds = model.predict(x)
    actual_prediction = np.argmax(preds) + 1
    return actual_prediction

random_output = model_predict(TRAIN_IMAGE_PATH, model)

#print(random_output)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)