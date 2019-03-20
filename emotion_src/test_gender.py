"""
test_gender dataset
"""
import numpy as np

from keras.models import load_model
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input

gender_model_path = '../trained_models/gender_models/gender_mini_XCEPTION.24-0.96.hdf5'
input_shape = (64, 64, 1)

def test_gender(dataset_name):
    print('Testing dataset: ', dataset_name)
    right = 0
    gender_classifier = load_model(gender_model_path, compile=False)

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    photos, genders = data_loader.get_data()
    photos = preprocess_input(photos)

    predictions = gender_classifier.predict(photos)
    predictions = np.argmax(predictions, axis = 1)
    real_gender = np.argmax(genders, axis = 1)
    num = len(predictions)
    for i in range(num):
        if predictions[i] == real_gender[i]:
            right += 1
        else:
            continue
    acc = (right / num) * 100
    # print(right)
    print('\033[0;31m%s\033[0m' % ('The acc is : %.2f%%' % acc))

# test_gender('KinFace')
test_gender('genki4k')