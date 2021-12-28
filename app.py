from flask import Flask, render_template

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'Models/VGG19_weights.h5'

model = load_model(MODEL_PATH)

print(model.summary())

def model_prediction(img_location, model):
    img = image.load_img(img_location, target_size = (224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis = 0)

    prediction = model.predict(x)
    prediction = np.argmax(prediction, axis = 1)
    return prediction


