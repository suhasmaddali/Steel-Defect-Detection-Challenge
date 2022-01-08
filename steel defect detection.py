import cv2
import pickle
import numpy as np
import sklearn
from imutils import paths
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

TEST_PATH = '../Steel Defect Detection Dataset/test_images'
TRAIN_PATH = '../Steel Defect Detection Dataset/train_images'
count = 0
for random_image in paths.list_images(TRAIN_PATH):
    if count == 0:
        img = cv2.imread(random_image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255
        break
    count += 1

img = np.array(img, dtype = np.float32)

img = img.reshape((1, 224, 224, 3))
pretrained_VGG19_model = load_model('Models/VGG19_weights.h5')
y_prediction = pretrained_VGG19_model.predict(img)
print(y_prediction)
class_prediction = np.argmax(y_prediction) + 1
print(class_prediction)

basepath = os.path.dirname(__file__)
file_path = os.path.join(basepath, 'uploads', secure_filename)