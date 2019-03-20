"""
File: train_RAF_classifier.py
Author: lvhui
Email:
Github:
Description: Train RAF emotion classifier model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION, tiny_XCEPTION
from models.mobilenet import MobileNetV3, self_model
from utils.datasets import DataManager
from utils.preprocessor import preprocess_input

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 100
base_path = '../trained_models/emotion_models/RAF/'

# data generator
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range =.1,
    horizontal_flip=True)

# model parameters/compilation
model = MobileNetV3(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

datasets = ['RAF']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callback
    log_file_path = base_path + dataset_name + '_emotion_mobilenetv3_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_mobilenet_v3'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    train_faces, train_emotions, test_faces, test_emotions = data_loader.get_data()
    train_faces = preprocess_input(train_faces)
    test_faces = preprocess_input(test_faces)
    num_samples, num_classes = train_emotions.shape
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(test_faces, test_emotions))