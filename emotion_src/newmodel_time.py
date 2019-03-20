import sys

import cv2
from keras.models import load_model
import numpy as np
import time

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import load_detection_model
from utils.inference import make_face_coordinates
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets

from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
image_path = sys.argv[1]
segment_model_path = '/home/tony/lvhui/process_fer2013/trained_models/align_model/fer2013/fer2013_unet.79-0.92.hdf5'
emotion_model_path = '../trained_models/emotion_models/fer2013/fer2013_mobilenet_V3.281-0.61.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (10, 10)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model()
segment_classifier = load_model(segment_model_path, compile=False)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

start = time.time()
# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

detected_faces, score, idx = detect_faces(face_detection, gray_image)

for detected_face in detected_faces:

    face_coordinates = make_face_coordinates(detected_face)


    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue


    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]

    color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

end = time.time()
print('{} s'.format((end-start)))

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/test.png',bgr_image)
img = cv2.imread('../images/test.png')
cv2.namedWindow('predicted_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('predicted_img', 880, 680)
cv2.imshow('predicted_img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()