import cv2
import pickle
import numpy as np
import sklearn
from imutils import paths
import matplotlib.pyplot as plt

TEST_PATH = '../Steel Defect Detection Dataset/test_images'
TEST_IMAGE = TEST_PATH + '/0000f269f.jpg'

plt.imread(TEST_IMAGE)
plt.imshow(TEST_IMAGE)
plt.show()