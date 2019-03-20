import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
image_path = sys.argv[1]
emotion_model_path = '../trained_models/emotion_models/CK+/CK+_mini_XCEPTION.138-1.00.hdf5'
emotion_labels = get_labels('CK+')
font = cv2.FONT_HERSHEY_SIMPLEX

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')


gray_face = cv2.resize(gray_image, (emotion_target_size))

gray_face = preprocess_input(gray_face, True)
gray_face = np.expand_dims(gray_face, 0)
gray_face = np.expand_dims(gray_face, -1)
emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
emotion_text = emotion_labels[emotion_label_arg]

print(emotion_text)