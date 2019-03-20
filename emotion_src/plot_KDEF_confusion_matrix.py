"""
plot confusion_matrix of KDEF
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

dataset_name = 'KDEF'
input_shape = (64, 64, 1)
validation_split = .2
save_path = '../trained_models/emotion_models/KDEF'

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def main():
    emotion_model_path = '../trained_models/emotion_models/KDEF/KDEF_mini_XCPTION.114-0.81.hdf5'
    emotion_classifier = load_model(emotion_model_path, compile=False)
    np.set_printoptions(precision=2)
    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, validation_split)
    val_faces, val_emotions = val_data
    predictions = emotion_classifier.predict(val_faces)
    predictions = np.argmax(predictions, axis=1)
    real_emotions = np.argmax(val_emotions, axis = 1)

    conf_mat = confusion_matrix(real_emotions, predictions)
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(conf_mat, classes=['Anger', 'Disgust', 'Afraid',
                                             'Happy', 'Sadness', 'Surprise', 'Neutral'],
                          normalize=True)
    plt.show()
    # plt.savefig(os.path.join(save_path, dataset_name+'Confusion Matrix.png'))
    # plt.close()

main()