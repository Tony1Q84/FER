from scipy.io import loadmat
from .normalize import HistEqualize
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
    imdb gender classification dataset."""
    def __init__(self, dataset_name = 'imdb', dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '/home/tony/lvhui/process_fer2013/datasets/fer2013_crop/'
            # self.dataset_path = '../datasets/fer2013/fer2013.csv'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = '../datasets/KDEF/'
        elif self.dataset_name == 'CK+':
            self.dataset_path = '../datasets/CK+/'
        elif self.dataset_name == 'jaffe':
            self.dataset_path = '../datasets/jaffe_crop/'
        elif self.dataset_name == 'RAF':
            self.dataset_path = '/home/tony/lvhui/process_fer2013/datasets/RAF/Image/RAF_crop_V2/'
        elif self.dataset_name == 'SFEW':
            self.dataset_path = '../datasets/SFEW/Train/'
        elif self.dataset_name == 'FERG_DB':
            self.dataset_path = '/home/tony/lvhui/self_face_emotion/datasets/FERG_DB/'
        elif self.dataset_name == 'KinFace':
            self.dataset_path = '../datasets/KinFace/'
        elif self.dataset_name == 'genki4k':
            self.dataset_path = '../datasets/genki4k/files/'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013_V2()
        elif self.dataset_name == 'CK+':
            ground_truth_data = self._load_CK()
        elif self.dataset_name == 'jaffe':
            ground_truth_data = self._load_jaffe()
        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()
        elif self.dataset_name == 'RAF':
            ground_truth_data = self._load_RAF()
        elif self.dataset_name == 'SFEW':
            ground_truth_data = self._load_SFEW()
        elif self.dataset_name == 'FERG_DB':
            ground_truth_data = self._load_FERG_DB()
        elif self.dataset_name == 'KinFace':
            ground_truth_data = self._load_KinFace()
        elif self.dataset_name == 'genki4k':
            ground_truth_data = self._load_genki4k()
        return ground_truth_data

    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            # face = illuminate(face)
            # histogram equalization
            # face = cv2.equalizeHist(face)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

    def _load_fer2013_V2(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape = (num_faces, y_size, x_size))
        emotions = np.zeros(shape = (num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            # image_array = HistEqualize(image_array)
            faces[file_arg] = image_array
            filepath, basename = os.path.split(file_path)
            filename, extension = os.path.splitext(basename)
            emotion = filename.split('_')[-1]
            emotion_arg = int(emotion)
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions

    def _load_CK(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.png')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[0:3]
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions

    def _load_jaffe(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith('.tiff'):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[3:5]
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions

    def _load_RAF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)
        emotion_label_path = '../datasets/RAF/EmoLabel/label.txt'
        label = {}
        with open(emotion_label_path) as labelline:
            for line in labelline:
                line = line.strip('\n')
                list = line.split(' ')
                label[list[0]] = int(list[1])-1

        train_face_paths = []
        test_face_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                line = filename.split('_')
                if line[0] == 'test':
                    test_face_paths.append(os.path.join(folder, filename))
                elif line[0] == 'train':
                    train_face_paths.append(os.path.join(folder, filename))

        train_num_faces = len(train_face_paths)
        test_num_faces = len(test_face_paths)
        y_size, x_size = self.image_size
        train_faces = np.zeros(shape=(train_num_faces, y_size, x_size))
        train_emotions = np.zeros(shape=(train_num_faces, num_classes))
        test_faces = np.zeros(shape=(test_num_faces, y_size, x_size))
        test_emotions = np.zeros(shape=(test_num_faces, num_classes))
        for file_arg, file_path in enumerate(train_face_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            train_faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = label[file_basename]
            train_emotions[file_arg, file_emotion] = 1
        train_faces = np.expand_dims(train_faces, -1)

        for file_arg, file_path in enumerate(test_face_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            test_faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = label[file_basename]
            test_emotions[file_arg, file_emotion] = 1
        test_faces = np.expand_dims(test_faces, -1)

        return train_faces, train_emotions, test_faces, test_emotions

    def _load_SFEW(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        test_base_path = '../datasets/SFEW/Val/'

        train_file_path = []
        test_file_path = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.png')):
                    train_file_path.append(os.path.join(folder, filename))

        train_num_faces = len(train_file_path)
        y_size, x_size  = self.image_size
        train_faces = np.zeros(shape=(train_num_faces, y_size, x_size))
        train_emotions = np.zeros(shape=(train_num_faces, num_classes))
        for file_arg, file_path in enumerate(train_file_path):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            train_faces[file_arg] = image_array
            folder_basename = file_path.split('/')[-2]
            file_emotion = folder_basename
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            train_emotions[file_arg, emotion_arg] = 1
        train_faces = np.expand_dims(train_faces, -1)

        for folder, subfoloders, filenames in os.walk(test_base_path):
            for filename in filenames:
                if filename.endswith(('.png')):
                    test_file_path.append(os.path.join(folder, filename))

        test_num_faces = len(test_file_path)
        y_size, x_size = self.image_size
        test_faces = np.zeros(shape=(test_num_faces, y_size, x_size))
        test_emotions = np.zeros(shape=(test_num_faces, num_classes))
        for file_arg, file_path in enumerate(test_file_path):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            test_faces[file_arg] = image_array
            folder_basename = file_path.split('/')[-2]
            file_emotion = folder_basename
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            test_emotions[file_arg, emotion_arg] = 1
        test_faces = np.expand_dims(test_faces, -1)

        return train_faces, train_emotions, test_faces, test_emotions

    def _load_FERG_DB(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.png')):
                    file_paths.append(os.path.join(folder, filename))

        # shuffle(file_paths)
        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            if os.path.exists(file_path):
                image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image_array = np.resize(image_array, (y_size, x_size))
                faces[file_arg] = image_array
                file_basename = os.path.basename(file_path)
                file_emotion = file_basename.split('_')[1]
                try:
                    emotion_arg = class_to_arg[file_emotion]
                except:
                    continue
                emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions

    def _load_KinFace(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_photo = len(file_paths)
        y_size, x_size= self.image_size
        photos = np.zeros(shape=(num_photo, y_size, x_size))
        genders = np.zeros(shape=(num_photo, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            photos[file_arg] = image_array
            photo_basename = os.path.basename(file_path)
            (photoname, extension) = os.path.splitext(photo_basename)
            gender = photoname.split('_')[1]
            try:
                gender_arg = class_to_arg[gender]
            except:
                continue
            genders[file_arg, gender_arg] = 1
        photos = np.expand_dims(photos, -1)
        return photos, genders

    def _load_genki4k(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        files_list = os.listdir(self.dataset_path)
        files_list.sort()
        for filename in files_list:
            if filename.endswith(('.jpg')):
                file_paths.append(os.path.join(self.dataset_path, filename))
        # for folder, subfolders, filenames in os.walk(self.dataset_path):
        #     for filename in filenames:
        #         if filename.endswith(('.jpg')):
        #             file_paths.append(os.path.join(folder, filename))

        gender_label_path = '../datasets/genki4k/gender_label.txt'
        gender_label = []
        with open(gender_label_path) as labelline:
            for line in labelline:
                gender_arg = int(line.strip('\n'))
                gender_label.append(gender_arg)

        num_photos = len(file_paths)
        y_size, x_size = self.image_size
        photos = np.zeros(shape=(num_photos, y_size, x_size))
        genders = np.zeros(shape=(num_photos, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            photos[file_arg] = image_array
            gender_arg = gender_label[file_arg]
            genders[file_arg, gender_arg] = 1
        photos = np.expand_dims(photos, -1)
        return photos, genders

    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            # there are two file names in the dataset that don't match the given classes
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions

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

# def  split_data(x, y, validation_split=.2):
#     num_samples = len(x)
#     num_train_samples = int((1 - validation_split)*num_samples)
#     train_x = x[:num_train_samples]
#     train_y = y[:num_train_samples]
#     val_x = x[num_train_samples:]
#     val_y = y[num_train_samples:]
#     train_data = (train_x, train_y)
#     val_data = (val_x, val_y)
#     return train_data, val_data

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