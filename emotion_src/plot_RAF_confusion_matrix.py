"""
plot confusion_matrix of RAF
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input

dataset_name = 'RAF'
input_shape = (64, 64, 1)
save_path = '../trained_models/emotion_models/RAF'

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title =None,
                          cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    :param cm:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest',  cmap=cmap)
    # plt.title(title, fontsize=10)
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

    # plt.ylabel('True label', fontsize=10)
    # plt.xlabel('Predicted label', fontsize=10)
    plt.tight_layout()

def main():
    true = 0
    emotion_model_path = '../trained_models/emotion_models/RAF/RAF_mini_XCEPTION.76-0.82.hdf5'
    emotion_classifier = load_model(emotion_model_path, compile=False)
    np.set_printoptions(precision=2)
    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    train_faces, train_emotions, test_faces, test_emotions = data_loader.get_data()
    test_faces = preprocess_input(test_faces)
    num_samples, num_classes = test_emotions.shape
    predictions = emotion_classifier.predict(test_faces)
    predictions = np.argmax(predictions, axis=1)
    real_emotion = np.argmax(test_emotions, axis = 1)
    # for i in range(num_samples):
    #     if predictions[i] == real_emotion[i]:
    #         true += 1
    # acc = 100. * true / num_samples
    conf_mat = confusion_matrix(real_emotion, predictions)
    plt.figure(figsize=(5, 4))
    plot_confusion_matrix(conf_mat, classes=['Sur', 'Fea', 'Dis', 'Hap',
                                             'Sad', 'Ang', 'Neu'],
                          normalize=True,
                         )
    # title='Normalized Confusion Matrix'
    plt.savefig(os.path.join(save_path, dataset_name + 'Confusion Matrix8-2.pdf'),  bbox_inches = 'tight', dpi = 600)
    plt.show()
    plt.close()

main()