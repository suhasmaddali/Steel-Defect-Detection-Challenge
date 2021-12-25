import cv2
import pickle
import numpy as np
import sklearn
from imutils import paths
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

TEST_PATH = '../Steel Defect Detection Dataset/test_images'
TEST_IMAGE = TEST_PATH + '/0000f269f.jpg'
count = 0
for random_image in paths.list_images(TEST_PATH):
    if count == 3:
        img = cv2.imread(random_image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255
    count += 1

img = np.array(img, dtype = np.float32)
print(img)
print(img.shape)
img = np.reshape(img, (1, 224, 224, 3))
pretrained_VGG19_model = load_model('Models/VGG19_weights.h5')
print(pretrained_VGG19_model)
print(pretrained_VGG19_model.summary())
y_prediction = pretrained_VGG19_model.predict(img)

print(y_prediction)
#plt.imread(TEST_IMAGE)
#plt.imshow(TEST_IMAGE)
#plt.show()