import cv2
import os
import numpy as np
from keras.models import load_model

from utils.grad_cam import compile_gradient_function
from utils.grad_cam import compile_saliency_function
from utils.grad_cam import register_gradient
from utils.grad_cam import modify_backprop
from utils.grad_cam import calculate_guided_gradient_CAM
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input
from utils.inference import load_image


color = (0, 255, 0)

def read_images(path):
    images = []
    for folder, subfolders, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(('jpg')):
                images.append(os.path.join(folder, filename))

    return images

datasets = ['fer2013']
for dataset_name in datasets:
    print('Processing :', dataset_name)

    global model_filename, image_path, labels, save_path, raw_path
    if dataset_name == 'fer2013':
        # labels = get_labels('fer2013')
        # offsets = (0, 0)
        # model_filename = '../trained_models/emotion_models/CK+/CK+_mini_XCEPTION.109-0.95.hdf5'
        model_filename = '../trained_models/emotion_models/fer2013/fer2013_tiny_xception.131-0.62.hdf5'
        image_path = '../images/test/fer2013/'
        save_path = '../images/test_save/fer2013/'
        raw_path = '../images/raw/fer2013/'
    elif dataset_name == 'RAF':
        # labels = get_labels('RAF')
        # offsets = (30, 60)
        model_filename = '../trained_models/emotion_models/RAF/RAF_tiny_XCEPTION.104-0.80.hdf5'
        image_path = '../images/test/RAF/'
        save_path = '../images/test_save/RAF/'
        raw_path = '../images/raw/RAF/'

    model = load_model(model_filename, compile=False)
    target_size = model.input_shape[1:3]
    images = read_images(image_path)

    for img in images:
        rgb_image = load_image(img, grayscale=False)
        gray_image = load_image(img, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_face = gray_image.astype('uint8')
        basename = os.path.basename(img)
        extension = str(basename.split('.')[0])
        save = save_path + extension + '.png'
        raw = raw_path + extension + '.png'

        try:
            gray_face = cv2.resize(gray_face, (target_size))
        except:
            continue

        cv2.imwrite(raw, gray_face)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        # prediction
        predicted_class = np.argmax(model.predict(gray_face))
        # label_text = labels[predicted_class]

        gradient_function = compile_gradient_function(model,
                                                      predicted_class, 'conv2d_7')

        # gradient_function = compile_gradient_function(model,

        #                                               predicted_class, 'block4_conv3')

        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp', model_filename)
        saliency_function = compile_saliency_function(guided_model, 'conv2d_7')
        # saliency_function = compile_saliency_function(guided_model, 'block4_conv3')

        guided_gradCAM = calculate_guided_gradient_CAM(gray_face,
                                                   gradient_function, saliency_function)
        guided_gradCAM = cv2.resize(guided_gradCAM, target_size)
        rgb_guided_gradCAM = np.repeat(guided_gradCAM[:, :, np.newaxis], 3, axis=2)

        cv2.imwrite(save, guided_gradCAM)
    print('Finished!')