"""
File: train_distdata_classifier.py
Author: lvhui
Email:
Github:
Description: Train emotion classifier model through landmark-distdance
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from models.cnn import mini_XCEPTION
# from utils.distance_data import DistanceManager
# from utils.distance_data import split_data
from utils.coordinate_data import CoordinateManager
from utils.coordinate_data import split_data
import numpy as np
import os

# parameters
batch_size = 64
num_epochs = 10000
input_shape = (68, 5, 1)
validation_split = .2
do_shuffle = True
verbose = 1
num_classes = 7
patience = 50
base_path = '../trained_models/emotion_models/'

def log_path(dataset_name):
    if dataset_name == 'fer2013':
        path = os.path.join(base_path, 'fer2013/')
    elif dataset_name == 'CK+':
        path = os.path.join(base_path, 'CK+/')
    elif dataset_name == 'KDEF':
        path = os.path.join(base_path, 'KDEF/')
    elif dataset_name == 'RAF':
        path = os.path.join(base_path, 'RAF/')
    elif dataset_name == 'SFEW':
        path = os.path.join(base_path, 'SFEW/')
    else:
        print('Please input right dataset name!')
    return path


# data generator
def data_generator(data, labels, shuffle = False):
    num_samples = len(data)
    steps_per_batch = int((num_samples - 1) / batch_size) + 1

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        shuffled_data = data[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]
    else:
        shuffled_data = data
        shuffled_labels = labels

    while True:
        for batch_num in range(steps_per_batch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, num_samples)
            X, Y = shuffled_data[start_index : end_index], shuffled_labels[start_index : end_index]
            yield X, Y

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 'fer2013', 'RAF', 'SFEW'
datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    path = log_path(dataset_name)
    # callback
    log_file_path = path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/5), verbose=1)
    trained_models_path = path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = CoordinateManager(dataset_name)
    dists, emotions = data_loader.get_data()
    # num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(dists, emotions, do_shuffle, validation_split)
    train_faces, train_emotions = train_data
    model.fit_generator(data_generator(train_faces, train_emotions,
                                            do_shuffle),
                        steps_per_epoch=int((len(train_faces) - 1) / batch_size) + 1,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)