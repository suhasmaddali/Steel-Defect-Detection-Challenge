from flask import Flask, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)


MODEL_PATH = 'Models/VGG19_weights.h5'

model = load_model(MODEL_PATH)

def model_prediction(img_location, model):
    img = image.load_img(img_location, target_size = (224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis = 0)

    prediction = model.predict(x)
    prediction = np.argmax(prediction, axis = 1)
    return prediction




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.jion(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_prediction(file_path, model)
        result = preds
        return result
    return None





if __name__ == '__main__':
    app.run(debug=True)