import os

CK_path = '../datasets/CK+/'

def rename():
    global new_name
    file_paths = []

    for folder, subfolders, filenames in os.walk(CK_path):
        for filename in filenames:
            if filename.endswith(('.png')):
                file_paths.append(os.path.join(folder, filename))

    for file_arg, file_path in enumerate(file_paths):
        folder_basename = file_path.split('/')[-2]
        file_basepath, file_basename = os.path.split(file_path)
        if folder_basename == 'anger':
            new_name = 'ang' + file_basename
        elif folder_basename == 'contempt':
            new_name = 'con' + file_basename
        elif folder_basename == 'disgust':
            new_name = 'dis' + file_basename
        elif folder_basename == 'fear':
            new_name = 'fea' + file_basename
        elif folder_basename == 'happy':
            new_name = 'hap' + file_basename
        elif folder_basename == 'sadness':
            new_name = 'sad' + file_basename
        elif folder_basename == 'surprise':
            new_name = 'sur' + file_basename

        os.rename(os.path.join(file_basepath, file_basename), os.path.join(file_basepath, new_name))


rename()