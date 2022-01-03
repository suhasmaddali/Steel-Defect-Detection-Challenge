from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

MODEL_PATH = 'Models/VGG19_weights.h5'

model = load_model(MODEL_PATH)

# model.summary()
TRAIN_IMAGE_PATH = '../Steel Defect Detection Dataset/train_images/00f6e702c.jpg'

img = image.load_img(TRAIN_IMAGE_PATH, target_size = (224, 224))
x = image.img_to_array(img)
x = x / 255
x = np.expand_dims(x, axis = 0)
preds = model.predict(x)
print(np.argmax(preds) + 1)