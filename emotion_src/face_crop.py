import os
import cv2
import numpy as np
# from utils.inference import load_image
from utils.inference import detect_faces
from utils.inference import load_detection_model
from utils.inference import make_face_coordinates

source_dataset_path = '../datasets/fer2013/'
cant_crop_path = '../datasets/fer2013_crop/cant_crop'

x_size = 48
y_size = 48
face_detection = load_detection_model()

image_paths = []
for folder, subfolders, filenames in os.walk(source_dataset_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder, filename))

num = len(image_paths)

for image_arg, image_path in enumerate(image_paths):
    print('Processing : {}/{}'.format(image_arg + 1, num))
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_image = np.squeeze(gray_image)
    # gray_image = gray_image.astype('uint8')
    detected_faces, score, idx = detect_faces(face_detection, gray_image)
    # for detected_face in detected_faces:
    if len(detected_faces) == 1:
        face_coordinates = make_face_coordinates(detected_faces)
        x, y, width, height = face_coordinates
        face = gray_image[y:y+height, x:x+width]
        face_image = cv2.resize(face, (y_size, x_size))
        source_path, filename = os.path.split(image_path)
        save_path = source_path.replace('fer2013', 'fer2013_crop')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, filename), face_image)
    else:
        source_path, filename = os.path.split(image_path)
        save_path = os.path.join(cant_crop_path, filename)
        cv2.imwrite(save_path, gray_image)


print('Finish!')