import csv
import numpy as np
import os
import cv2
from skimage import io
from face_alignment.api import FaceAlignment, LandmarksType

fa_2D = FaceAlignment(LandmarksType._2D, flip_input=False)
fa_3D = FaceAlignment(LandmarksType._3D, flip_input=False)


class DataTransform(object):
    """"
    class for get the landmarks of dataset and tranform
    it to csv
    """
    def __init__(self, dataset_name = 'fer2013', dataset_path = None, method = '3D'):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.method = method
        # if self.method == '2D':
        #     self.landmark_detect = FaceAlignment(LandmarksType._2D, flip_input=False)
        # elif self.method == '3D':
        #     self.landmark_detect = FaceAlignment(LandmarksType._3D, flip_input=False)
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/'
        elif self.dataset_name == 'KDEF':
            self.dataset_path = '../datasets/KDEF/'
        elif self.dataset_name == 'CK+':
            self.dataset_path = '../datasets/CK+/'
        elif self.dataset_name == 'RAF':
            self.dataset_path = '../datasets/RAF/Image/'
        elif self.dataset_name == 'SFEW':
            self.dataset_path = '../datasets/SFEW/Train/'
        else:
            raise Exception('Incorrect dataset name, please input right datasetname!')

    def get_data(self):
        if self.dataset_name == 'fer2013':
            transform_data = self._load_fer2013()
        elif self.dataset_name == 'KDEF':
            transform_data = self._load_KDEF()
        elif self.dataset_name == 'CK+':
            transform_data = self._load_CK()
        elif self.dataset_name == 'RAF':
            transform_data = self._load_RAF()
        elif self.dataset_name == 'SFEW':
            transform_data = self._load_SFEW()

        return transform_data

    # def _load_fer2013(self):
    #     headers = ['filename', 'emotion', 'distarray', 'Usage']
    #     rows = []
    #     with open(self.dataset_path, 'r') as csvin:
    #         data = csv.reader(csvin)
    #         for line_arg, row in enumerate(data):
    #             detail = {}
    #             detail['file_arg'] = line_arg
    #             detail['emotion'] = row[0]
    #             if row[-1] != 'Usage':
    #                 face = []
    #                 for pixel in row[1].split(' '):
    #                     face.append(int(pixel))
    #                 face = np.asarray(face).reshape(width, height)
    #                 face = face.astype('float32')
    #                 landmarks = self.landmark_detect.get_landmarks(face)
    #                 if landmarks == None:
    #                     continue
    #                 distance = self.calculate_distance(landmarks)
    #                 detail['distarray'] = distance
    #             detail['Usage'] = row[-1]
    #             rows.append(detail)
    #
    #     print('Finish transforming!')
    #     return headers, rows

    def _load_fer2013(self):
        # 1203 non_face, 10 more_face
        headers = ['filename', 'emotion', 'distarray', 'Usage']
        rows = []
        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append((os.path.join(folder, filename)))

        total = len(file_paths)
        non_face = 0
        more_face = 0

        for file_arg, file_path in enumerate(file_paths):
            print("Processing: {}/{}".format(file_arg, total))
            details = {}
            file_basename = os.path.basename(file_path)
            details['filename'] = file_basename
            file_emotion = file_path.split('/')[-2]
            details['emotion'] = int(file_emotion)

            image_array = io.imread(file_path)
            landmarks = fa_2D.get_landmarks(image_array)
            if landmarks is not None:
                landmarks = np.asarray(landmarks)
                if landmarks.shape  == (1, 68, 2):
                    landmarks = np.squeeze(landmarks)
                    distance = self.calculate_distance(landmarks)
                    store_str = array_to_str(distance)
                    details['distarray'] = store_str
                else:
                    more_face += 1
                    continue
            else:
                non_face += 1
                continue

            usage = file_path.split('/')[-3]
            details['Usage'] = usage
            rows.append(details)

        return headers, rows, non_face, more_face

    def _load_KDEF(self):
        # 10 non_face 6 more_face
        headers = ['filename', 'emotion', 'distarray']
        rows = []
        class_to_arg = get_class_to_arg(self.dataset_name)
        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append((os.path.join(folder, filename)))

        total = len(file_paths)
        non_face = 0
        more_face = 0

        for file_arg, file_path in enumerate(file_paths):
            print('Processing: {}/{}'.format(file_arg, total))
            details = {}
            file_basename = os.path.basename(file_path)
            details['filename'] = file_basename
            file_emotion = file_basename[4:6]
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            details['emotion'] = emotion_arg

            image_array = io.imread(file_path)
            landmarks = fa_2D.get_landmarks(image_array)
            if landmarks is not None:
                landmarks = np.asarray(landmarks)
                if landmarks.shape == (1, 68, 2):
                    landmarks = np.squeeze(landmarks)
                    distance = self.calculate_distance(landmarks)
                    store_str = array_to_str(distance)
                    details['distarray'] = store_str
                else:
                    more_face += 1
                    continue
            else:
                non_face += 1
                continue
            rows.append(details)

        return headers, rows, non_face, more_face

    def _load_CK(self):
        # 0 non_face 0 more_face
        class_to_arg = get_class_to_arg(self.dataset_name)
        header = ['filename', 'emotion', 'distarray']
        rows = []
        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('.png')):
                    file_paths.append(os.path.join(folder, filename))

        total = len(file_paths)
        non_face = 0
        more_face = 0

        for file_arg, file_path in enumerate(file_paths):
            print('Processing: {}/{}'.format(file_arg, total))
            details = {}
            file_basename = os.path.basename(file_path)
            details['filename'] = file_basename
            file_emotion = file_basename[0:3]
            emotion_arg = class_to_arg[file_emotion]
            details['emotion'] = emotion_arg

            image_array = io.imread(file_path)
            landmarks = fa_2D.get_landmarks(image_array)
            if landmarks is not None:
                landmarks = np.asarray(landmarks)
                if landmarks.shape == (1, 68, 2):
                    landmarks = np.squeeze(landmarks)
                    distance = self.calculate_distance(landmarks)
                    store_str = array_to_str(distance)
                    details['distarray'] = store_str
                else:
                    more_face += 1
                    continue
            else:
                non_face += 1
                continue
            rows.append(details)

        return header, rows, non_face, more_face

    def _load_RAF(self):
        # 114 non_face 7 more_face
        header = ['filename', 'emotion', 'distarray', 'Usage']
        rows = []
        emotion_label_path = '../datasets/RAF/EmoLabel/label.txt'
        labels = {}
        with open(emotion_label_path) as labelline:
            for line in labelline:
                line = line.strip(('\n'))
                list = line.split(' ')
                labels[list[0]] = int(list[1]) - 1

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('jpg')):
                    file_paths.append(os.path.join(folder, filename))

        total =  len(file_paths)
        non_face = 0
        more_face = 0

        for file_arg, file_path in enumerate(file_paths):
            print('Processing : {}/{}'.format(file_arg, total))
            details = {}
            file_basename = os.path.basename(file_path)
            details['filename'] = file_basename
            file_emotion = labels[file_basename]
            details['emotion'] = file_emotion

            image_array = io.imread(file_path)
            landmarks = fa_2D.get_landmarks(image_array)
            if landmarks is not None:
                landmarks = np.asarray(landmarks)
                if landmarks.shape == (1, 68, 2):
                    landmarks = np.squeeze(landmarks)
                    distance = self.calculate_distance(landmarks)
                    store_str = array_to_str(distance)
                    details['distarray'] = store_str
                else:
                    more_face += 1
                    continue
            else:
                non_face += 1
                continue

            line = file_basename.split('_')
            if line[0] == 'test':
                details['Usage'] = 'Test'
            elif line[0] == 'train':
                details['Usage'] = 'Train'

            rows.append(details)

        return header, rows, non_face, more_face

    def _load_SFEW(self):
        # 24 non_face 0 more_face
        class_to_arg = get_class_to_arg(self.dataset_name)
        test_base_path = '../datasets/SFEW/Val/'
        header = ['filename', 'emotion', 'distarray', 'Usage']
        rows = []

        train_file_path = []
        test_file_path = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(('png')):
                    train_file_path.append(os.path.join(folder, filename))

        for folder, subfolders, filenames in os.walk(test_base_path):
            for filename in filenames:
                if filename.endswith(('.png')):
                    test_file_path.append(os.path.join(folder, filename))

        train_total = len(train_file_path)
        test_total = len(test_file_path)
        train_non_face = 0
        train_more_face = 0
        test_non_face = 0
        test_more_face = 0

        for file_arg, file_path in enumerate(train_file_path):
            print('Processing : {}/{}'.format(file_arg, train_total))
            details = {}
            file_basename = os.path.basename(file_path)
            details['filename'] = file_basename
            folder_basename = file_path.split('/')[-2]
            file_emotion = folder_basename
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            details['emotion'] = emotion_arg

            image_array = io.imread(file_path)
            landmarks = fa_2D.get_landmarks(image_array)
            if landmarks is not None:
                landmarks = np.asarray(landmarks)
                if landmarks.shape == (1, 68, 2):
                    landmarks = np.squeeze(landmarks)
                    distance = self.calculate_distance(landmarks)
                    store_str = array_to_str(distance)
                    details['distarray'] = store_str
                else:
                    train_more_face += 1
                    continue
            else:
                train_non_face += 1
                continue
            details['Usage'] = 'Train'

            rows.append(details)

        for file_arg, file_path in enumerate(test_file_path):
            print('Processing : {}/{}'.format(file_arg, test_total))
            details = {}
            file_basename = os.path.basename(file_path)
            details['filename'] = file_basename
            folder_basename = file_path.split('/')[-2]
            file_emotion = folder_basename
            emotion_arg = class_to_arg[file_emotion]
            details['emotion'] = emotion_arg

            image_array = io.imread(file_path)
            landmarks = fa_2D.get_landmarks(image_array)
            if landmarks is not None:
                landmarks = np.asarray(landmarks)
                if landmarks.shape == (1, 68, 2):
                    landmarks = np.squeeze(landmarks)
                    distance = self.calculate_distance(landmarks)
                    store_str = array_to_str(distance)
                    details['distarray'] = store_str
                else:
                    test_more_face += 1
                    continue
            else:
                test_non_face += 1
                continue
            details['Usage'] = 'Test'

            rows.append(details)

        non_face = train_non_face + test_non_face
        more_face = train_more_face + test_more_face

        return header, rows, non_face, more_face


    # def write_to_csv(self):
    #     headers =[]
    #     rows = []
    #     save_path = ''
    #     if self.dataset_name == 'fer2013':
    #         headers, rows = self._load_fer2013()
    #         save_path = '../datasets/fer2013/'
    #     elif self.dataset_name == 'KDEF':
    #         headers, rows = self._load_KDEF()
    #         save_path = '../datasets/KDEF/'
    #     elif self.dataset_name == 'CK+':
    #         headers, rows = self._load_CK()
    #         save_path = '../datasets/CK+/'
    #     elif self.dataset_name == 'RAF':
    #         headers, rows = self._load_RAF()
    #         save_path = '../datasets/RAF/'
    #     elif self.dataset_name == 'SFEW':
    #         headers, rows = self._load_SFEW()
    #         save_path = '../datasets/SFEW/'
    #
    #     csv_path = os.path.join(save_path, 'data.csv')
    #     with open(csv_path, 'w') as f:
    #         f_csv = csv.DictWriter(f, headers)
    #         f_csv.writeheader()
    #         f_csv.writerows(rows)
    #     print('Finish writing!')



    def calculate_distance(self, face_landmark):
        w = face_landmark.shape[0]
        h = w
        distance = np.zeros((h, w))
        for i in range(face_landmark.shape[0]):
            for j in range(face_landmark.shape[0]):
                dist = np.linalg.norm(np.asarray(face_landmark[i])
                                      - np.asarray(face_landmark[j]))
                distance[i][j] = float('%.2f' % dist)

        return distance

def array_to_str(distance):
    distance = distance.flatten()
    store_str = ''
    for i in range(len(distance)):
        store_str = store_str + ' ' + str(distance[i])
    return store_str

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry', 1:'disgust', 2:'fear', 3:'happy',
                4:'sad', 5:'surprise', 6:'neutral'}

    elif dataset_name == 'CK+':
        return {0:'ang', 1:'con', 2:'dis', 3:'fea',
                4:'hap', 5:'sad', 6:'sur'}

    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}

    elif dataset_name == 'RAF':
        return {0:'Surprise', 1:'Fear', 2:'Disgust', 3:'Happiness',
                4:'Sadness', 5:'Anger', 6:'Neutral'}

    elif dataset_name == 'SFEW':
        return {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
                4:'Neutral', 5:'Sad', 6:'Surprise'}

    else:
        raise Exception('Invalid dataset name')

def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                'surprise':5, 'neutral':6}

    elif dataset_name == 'CK+':
        return {'ang':0, 'con':1, 'dis':2, 'fea':3,
                'hap':4, 'sad':5, 'sur':6}

    elif dataset_name == 'KDEF':
        return {'AN':0, 'DI':1, 'AF':2, 'HA':3, 'SA':4, 'SU':5, 'NE':6}

    elif dataset_name == 'RAF':
        return {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happiness':3, 'Sadness':4,
                'Anger':5, 'Neutral':6}

    elif dataset_name == 'SFEW':
        return {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3,
                'Neutral':4, 'Sad':5, 'Surprise':6}

    else:
        raise Exception('Invalid dataset name')

def write_to_csv(headers, rows, dataset_name):
    save_path =''
    if dataset_name == 'fer2013':
        # headers, rows = self._load_fer2013()
        save_path = '../datasets/fer2013/'
    elif dataset_name == 'KDEF':
        # headers, rows = self._load_KDEF()
        save_path = '../datasets/KDEF/'
    elif dataset_name == 'CK+':
        # headers, rows = self._load_CK()
        save_path = '../datasets/CK+/'
    elif dataset_name == 'RAF':
        # headers, rows = self._load_RAF()
        save_path = '../datasets/RAF/'
    elif dataset_name == 'SFEW':
        # headers, rows = self._load_SFEW()
        save_path = '../datasets/SFEW/'

    csv_path = os.path.join(save_path, '2D_two_data.csv')
    with open(csv_path, 'w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(rows)
    print('Finish writing!')

