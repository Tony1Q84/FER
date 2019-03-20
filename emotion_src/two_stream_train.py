import numpy as np
import os
import cv2
import csv
from utils.coordinate_data import one_hot
from utils.coordinate_data import mean_input
from utils.normalize import illuminate
from utils.preprocessor import preprocess_input
from models.Two_stream_CNN import two_stream
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau


# parameters
batch_size = 32
num_epochs = 10000
coor_shape = (68, 5)
img_shape = (48, 48, 1)
do_shuffle = True
verbose = 1
num_classes = 7
patience = 50
image_base_path = '../datasets/fer2013/'
coor_base_path = '../datasets/fer2013/dist_angle.csv'
log_path = '../trained_models/emotion_models/fer2013/two_stream/'

def getdata(coor_path):
    coors = []
    imgs = []
    emotions = []

    with open(coor_path, 'r') as csvin:
        data = csv.reader(csvin)
        for line_arg, row in enumerate(data):
            usage = row[-1]
            basename = row[0]
            emotion = row[1]
            if usage == 'Training' or usage == 'PrivateTest' or usage == 'PublicTest':
                coor_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                coor_array = np.asarray(coor_list).reshape(68, 5)
                coor_array = mean_input(coor_array)
                coors.append(coor_array)
                filename = usage + '/' + emotion + '/' + basename
                img_path = os.path.join(image_base_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img.astype('uint8'), (48, 48))
                img_array = illuminate(img_array)
                imgs.append(img_array)
                emotions.append(int(emotion))
            else:
                continue
    coors = np.asarray(coors)
    # coors = np.expand_dims(coors, -1)
    imgs = np.asarray(imgs)
    imgs = np.expand_dims(imgs, -1)
    emotions = np.asarray(emotions)
    emotions = one_hot(emotions, 7)

    return imgs, coors, emotions

def  split_data(img_data, coor_data, emotion, do_shuffle = True, validation_split=.2):
    num_samples = len(img_data)

    if do_shuffle == True:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        shuffled_img = img_data[shuffle_indices]
        shuffled_coor = coor_data[shuffle_indices]
        shuffled_emotion = emotion[shuffle_indices]
    else:
        shuffled_img = img_data
        shuffled_coor = coor_data
        shuffled_emotion = emotion

    num_train_samples = int((1 - validation_split)*num_samples)
    train_img = shuffled_img[:num_train_samples]
    train_coor = shuffled_coor[:num_train_samples]
    train_emotion = shuffled_emotion[:num_train_samples]
    test_img = shuffled_img[num_train_samples:]
    test_coor = shuffled_coor[num_train_samples:]
    test_emotion = shuffled_emotion[num_train_samples:]
    train_data = (train_img, train_coor, train_emotion)
    test_data = (test_img, test_coor, test_emotion)
    return train_data, test_data

def data_generator(img_data, coor_data, emotion, shuffle = True):
    num_samples = len(img_data)
    steps_per_batch = int((num_samples - 1) / batch_size) + 1

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        shuffled_img = img_data[shuffle_indices]
        shuffled_coor = coor_data[shuffle_indices]
        shuffled_emotion = emotion[shuffle_indices]
    else:
        shuffled_img = img_data
        shuffled_coor = coor_data
        shuffled_emotion = emotion

    while True:
        for batch_num in range(steps_per_batch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, num_samples)
            X, Y, Z= shuffled_img[start_index : end_index], \
                   shuffled_coor[start_index : end_index], shuffled_emotion[start_index : end_index]
            yield [X, Y], Z

# x, y, z = getdata(coor_base_path)
# print('test_img_shape: ', x.shape)
# print('test_coor_shape: ', y.shape)
# print('test_emotion_shape: ', z.shape)

# model parameters/compilation
model = two_stream(img_shape, coor_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 'fer2013', 'RAF', 'SFEW'
datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callback
    log_file_path = log_path + dataset_name + '_emotion_trainingV2.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/5), verbose=1)
    trained_models_path = log_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    imgs, coors, emotions = getdata(coor_base_path)
    train_data, test_data = split_data(imgs, coors, emotions)
    train_img, train_coor, train_emotion = train_data
    test_img, test_coor, test_emotion = test_data
    train_img = preprocess_input(train_img)
    test_img = preprocess_input(test_img)
    model.fit_generator(data_generator(train_img, train_coor, train_emotion, do_shuffle),
                        steps_per_epoch=int((len(train_img) - 1) / batch_size) + 1,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=([test_img, test_coor], test_emotion))