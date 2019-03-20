import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2
import csv

class DistanceManager(object):
    """Class for loading fer2013 emotion classification dataset or
    imdb gender classification dataset."""
    def __init__(self, dataset_name = 'fer2013', dataset_path=None):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/2D_data.csv'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = '../datasets/KDEF/2D_data.csv'
        elif self.dataset_name == 'CK+':
            self.dataset_path = '../datasets/CK+/2D_data.csv'
        elif self.dataset_name == 'RAF':
            self.dataset_path = '../datasets/RAF/2D_data.csv'
        elif self.dataset_name == 'SFEW':
            self.dataset_path = '../datasets/SFEW/2D_data.csv'
        else:
            raise Exception('Incorrect dataset name, please input right dataset name')

    def get_data(self):
        if self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'CK+':
            ground_truth_data = self._load_CK()
        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()
        elif self.dataset_name == 'RAF':
            ground_truth_data = self._load_RAF_V2()
        elif self.dataset_name == 'SFEW':
            ground_truth_data = self._load_SFEW_V2()
        return ground_truth_data

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        distlist = data['distarray'].tolist()
        width, height = 68, 68
        distances_array = []
        for dist_sequence in distlist:
            dist_list = [np.float32(num) for num in dist_sequence.strip().split(' ')]
            # max_value = max(dist_list)
            dist_array = np.asarray(dist_list).reshape(width, height)
            dist_array = preprocess_input_V2(dist_array)
            distances_array.append(dist_array)
        distances_array = np.asarray(distances_array)
        dists = np.expand_dims(distances_array, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        return dists, emotions


    def _load_fer2013_V2(self):
        width, height = 68, 68
        train_dists = []
        train_emotions = []
        test_dists = []
        test_emotions = []
        with open(self.dataset_path, 'r') as csvin:
            data = csv.reader(csvin)
            for line_arg, row in enumerate(data):
                usage = row[-1]
                if usage == 'Training':
                    dist_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                    dist_array = np.asarray(dist_list).reshape(width, height)
                    dist_array = preprocess_input_V2(dist_array)
                    train_dists.append(dist_array)
                    train_emotions.append(int(row[1]))
                elif usage == 'PublicTest' or usage == 'PrivateTest':
                    dist_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                    dist_array = np.asarray(dist_list).reshape(width, height)
                    dist_array = preprocess_input_V2(dist_array)
                    test_dists.append(dist_array)
                    test_emotions.append(int(row[1]))
        train_dists = np.asarray(train_dists)
        train_dists = np.expand_dims(train_dists, -1)
        train_emotions = np.asarray(train_emotions)
        train_emotions = one_hot(train_emotions, 7)
        test_dists = np.asarray(test_dists)
        test_dists = np.expand_dims(test_dists, -1)
        test_emotions = np.asarray(test_emotions)
        test_emotions = one_hot(test_emotions, 7)

        return train_dists, train_emotions, test_dists, test_emotions


    def _load_CK(self):
        data = pd.read_csv(self.dataset_path)
        distlist = data['distarray'].tolist()
        width, height = 68, 68
        distances_array = []
        for dist_sequence in distlist:
            dist_list = [np.float32(num) for num in dist_sequence.strip().split(' ')]
            dist_array = np.asarray(dist_list).reshape(width, height)
            distances_array.append(dist_array)
        distances_array = np.asarray(distances_array)
        dists = np.expand_dims(distances_array, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return dists, emotions

    def _load_RAF(self):
        data = pd.read_csv(self.dataset_path)
        distlist = data['distarray'].tolist()
        width, height = 68, 68
        distances_array = []
        for dist_sequence in distlist:
            dist_list = [np.float32(num) for num in dist_sequence.strip().split(' ')]
            dist_array = np.asarray(dist_list).reshape(width, height)
            distances_array.append(dist_array)
        distances_array = np.asarray(distances_array)
        dists = np.expand_dims(distances_array, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        return dists, emotions

    def _load_RAF_V2(self):
        width, height = 68, 68
        train_dists = []
        train_emotions = []
        test_dists = []
        test_emotions = []
        with open(self.dataset_path, 'r') as csvin:
            data = csv.reader(csvin)
            for line_arg, row in enumerate(data):
                usage = row[-1]
                if usage == 'Train':
                    dist_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                    dist_array = np.asarray(dist_list).reshape(width, height)
                    train_dists.append(dist_array)
                    train_emotions.append(int(row[1]))
                elif usage == 'Test':
                    dist_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                    dist_array = np.asarray(dist_list).reshape(width, height)
                    test_dists.append(dist_array)
                    test_emotions.append(int(row[1]))
        train_dists = np.asarray(train_dists)
        train_dists = np.expand_dims(train_dists, -1)
        train_emotions = np.asarray(train_emotions)
        train_emotions = one_hot(train_emotions, 7)
        test_dists = np.asarray(test_dists)
        test_dists = np.expand_dims(test_dists, -1)
        test_emotions = np.asarray(test_emotions)
        test_emotions = one_hot(test_emotions, 7)

        return train_dists, train_emotions, test_dists, test_emotions

    def _load_SFEW(self):
        data = pd.read_csv(self.dataset_path)
        distlist = data['distarray'].tolist()
        width, height = 68, 68
        distances_array = []
        for dist_sequence in distlist:
            dist_list = [np.float32(num) for num in dist_sequence.strip().split(' ')]
            dist_array = np.asarray(dist_list).reshape(width, height)
            distances_array.append(dist_array)
        distances_array = np.asarray(distances_array)
        dists = np.expand_dims(distances_array, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        return dists, emotions

    def _load_SFEW_V2(self):
        width, height = 68, 68
        train_dists = []
        train_emotions = []
        test_dists = []
        test_emotions = []
        with open(self.dataset_path, 'r') as csvin:
            data = csv.reader(csvin)
            for line_arg, row in enumerate(data):
                usage = row[-1]
                if usage == 'Train':
                    dist_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                    dist_array = np.asarray(dist_list).reshape(width, height)
                    train_dists.append(dist_array)
                    train_emotions.append(int(row[1]))
                elif usage == 'Test':
                    dist_list = [np.float32(train_num) for train_num in row[2].strip().split(' ')]
                    dist_array = np.asarray(dist_list).reshape(width, height)
                    test_dists.append(dist_array)
                    test_emotions.append(int(row[1]))
        train_dists = np.asarray(train_dists)
        train_dists = np.expand_dims(train_dists, -1)
        train_emotions = np.asarray(train_emotions)
        train_emotions = one_hot(train_emotions, 7)
        test_dists = np.asarray(test_dists)
        test_dists = np.expand_dims(test_dists, -1)
        test_emotions = np.asarray(test_emotions)
        test_emotions = one_hot(test_emotions, 7)

        return train_dists, train_emotions, test_dists, test_emotions

    def _load_KDEF(self):
        data = pd.read_csv(self.dataset_path)
        distlist = data['distarray'].tolist()
        width, height = 68, 68
        distances_array = []
        for dist_sequence in distlist:
            dist_list = [np.float32(num) for num in dist_sequence.strip().split(' ')]
            dist_array = np.asarray(dist_list).reshape(width, height)
            distances_array.append(dist_array)
        distances_array = np.asarray(distances_array)
        dists = np.expand_dims(distances_array, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        return dists, emotions

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry', 1:'disgust', 2:'fear', 3:'happy',
                4:'sad', 5:'surprise', 6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'Kinface':
        return {0:'female', 1:'male'}
    elif dataset_name == 'genki4k':
        return {0:'women', 1:'man'}
    elif dataset_name == 'CK+':
        return {0:'ang', 1:'con', 2:'dis', 3:'fea',
                4:'hap', 5:'sad', 6:'sur'}
    elif dataset_name == 'jaffe':
        return {0:'AN', 1:'DI', 2:'FE', 3:'HA', 4:'NE', 5:'SA', 6:'SU'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    elif dataset_name == 'RAF':
        return {0:'Surprise', 1:'Fear', 2:'Disgust', 3:'Happiness',
                4:'Sadness', 5:'Anger', 6:'Neutral'}
    elif dataset_name == 'SFEW':
        return {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
                4:'Neutral', 5:'Sad', 6:'Surprise'}
    elif dataset_name == 'FERG_DB':
        return {0:'anger', 1:'disgust', 2:'fear', 3:'joy',
                4:'neutral', 5:'sadness', 6:'surprise'}
    else:
        raise Exception('Invalid dataset name')

def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}
    elif dataset_name == 'imdb':
        return {'women':0, 'man':1}
    elif dataset_name == 'KinFace':
        return {'female':0, 'male':1}
    elif dataset_name == 'genki4k':
        return {'women':0, 'man':1}
    elif dataset_name == 'CK+':
        return {'ang':0, 'con':1, 'dis':2, 'fea':3,
                'hap':4, 'sad':5, 'sur':6}
    elif dataset_name == 'jaffe':
        return {'AN':0, 'DI':1, 'FE':2, 'HA':3, 'NE':4, 'SA':5, 'SU':6}
    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}
    elif dataset_name == 'RAF':
        return {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happiness':3, 'Sadness':4,
                'Anger':5, 'Neutral':6}
    elif dataset_name == 'SFEW':
        return {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3,
                'Neutral':4, 'Sad':5, 'Surprise':6}
    elif dataset_name == 'FERG_DB':
        return {'anger':0, 'disgust':1, 'fear':2, 'joy':3,
                'neutral':4, 'sadness':5, 'surprise':6}
    else:
        raise Exception('Invalid dataset name')

def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def  split_data(x, y, do_shuffle = True, validation_split=.2):
    num_samples = len(x)

    if do_shuffle == True:
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        shuffled_data = x[shuffle_indices]
        shuffled_labels = y[shuffle_indices]
    else:
        shuffled_data = x
        shuffled_labels = y

    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = shuffled_data[:num_train_samples]
    train_y = shuffled_labels[:num_train_samples]
    val_x = shuffled_data[num_train_samples:]
    val_y = shuffled_labels[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

def one_hot(arr, num_classes=None):
    arr = np.array(arr, dtype='int')
    input_shape = arr.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    arr = arr.ravel()
    if not num_classes:
        num_classes = np.max(arr) + 1
    n = arr.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), arr] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical

def preprocess_input(x, max_value, v2=True):
    # x = x.astype('float32')
    max_value = float(max_value)
    x = x / max_value
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def preprocess_input_V2(x):

    mean_value = np.mean(x)
    std_vaue = np.std(x)
    x = (x - mean_value) / std_vaue
    # if v2:
    #     x = x - 0.5
    #     x = x * 2.0
    return x

