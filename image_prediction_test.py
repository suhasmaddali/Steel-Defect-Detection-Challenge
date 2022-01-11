from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
MODEL_PATH = 'Models/VGG19_weights.h5'

model = load_model(MODEL_PATH)
app = Flask(__name__)
# model.summary()
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


@app.route('/', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(basepath, 'uploads')
        f.save(os.path.join(file_path, secure_filename(f.filename)))
        print(file_path)
        predictions = model_predict(file_path, model)
        result = predictions
        #print(result)
        result = str(result)
        return result
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)