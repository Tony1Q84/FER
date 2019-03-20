"""
File: train_CK_classifier.py
Author: Tony
Email:
Github:
Description: Train CK+ emotion classification model
"""
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION
from utils.datasets import DataManager
# from utils.datasets import split_data
from utils.preprocessor import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# parameters
batch_size = 32
num_epochs = 100
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 100
base_path = '../trained_models/emotion_models/CK+/'

# data generator
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()


datasets = ['CK+']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience= int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    print(faces.shape, emotions.shape)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     faces, emotions, test_size=0.4, random_state=0
    # )
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)
    faces = preprocess_input(faces)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cvscores = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(faces, np.argmax(emotions, axis=1))):
        print('\nFold ', i)
        faces_train = faces[train_idx]
        emotions_train = emotions[train_idx]
        faces_valid = faces[val_idx]
        emotions_valid = emotions[val_idx]
        model.fit_generator(data_generator.flow(faces_train, emotions_train,
                                                batch_size),
                            steps_per_epoch=len(faces_train)/ batch_size,
                            epochs=num_epochs, shuffle=True, verbose=1, callbacks=callbacks,
                            validation_data=(faces_valid, emotions_valid))
        scores = model.evaluate(faces_valid, emotions_valid)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1]*100)
    print("%.2f%% +/- %.2f%%" % (np.mean(cvscores), np.std(cvscores)))