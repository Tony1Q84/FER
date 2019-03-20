from face_alignment.api import FaceAlignment, LandmarksType
# from skimage import io
# import cv2
# import numpy as np
#
# fa = FaceAlignment(LandmarksType._3D, flip_input=False)
#
# input = io.imread('/home/tony/lvhui/self_face_emotion/datasets/RAF/Image/aligned/train_09493_aligned.jpg')
#
# preds = fa.get_landmarks(input)
#
# preds = np.asarray(preds)
# if preds is not None:
#     print(preds)
#     print(preds.shape)
# else:
#     print('No lanmark!')
#
#
# import numpy as np
#
# def calculate_distance(face_landmark):
#     w = face_landmark.shape[0]
#     h = w
#     distance = np.zeros((h, w))
#     for i in range(face_landmark.shape[0]):
#         for j in range (face_landmark.shape[0]):
#             dist = np.linalg.norm(np.asarray(face_landmark[i])
#                                             - np.asarray(face_landmark[j]))
#             distance[i][j] = round(dist, 2)
#
#     return distance
#
# dist = calculate_distance(preds)
# print(dist)
# print(dist.shape)

# from dataset_to_csv import DataTransform, write_to_csv
from coordinate_to_csv import Coordinate, write_to_csv

# 'fer2013', 'RAF', 'SFEW', 'CK+'
datasets = ['fer2013']
for dataset_name in datasets:
    print('Transforming : ', dataset_name)

    # TransForm = DataTransform(dataset_name)
    # headers, rows, non_face, more_face = TransForm.get_data()

    TransForm = Coordinate(dataset_name)
    headers, rows, non_face, more_face = TransForm.get_data()

    print('There is {} faces not detected!'.format(non_face))
    print('There is {} photos has more than one face!'.format(more_face))
    print('Start writing!')
    write_to_csv(headers, rows, dataset_name)

    print('Transformed:', dataset_name)