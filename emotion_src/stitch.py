import os
from PIL import Image


def read_images(path):
    images = []
    for folder, subfolders, filenames in os.walk(path):
        for filename in filenames:
            images.append(os.path.join(folder, filename))

def Horizontal_stiching(images, unit_size):
    length = len(images)
    target_width = unit_size * length
    imagefile = []
    for img in range(length):
        imagefile.append(Image.open(images[img]))
    target = Image.new('RGB', (target_width, unit_size))
    left = 0
    right = unit_size
    for image in imagefile:
        target.paste(image, (left, 0, right, unit_size))
        left += unit_size
        right +=unit_size
    quality_value = 100
    target.save('result.png', quality_value = quality_value)

    print('Done!')

def Vertival_stiching(images, unit_size):
    length = len(images)
    target_height = unit_size * length
    imagefile = []
    for img in range(length):
        imagefile.append(Image.open(img))
    target = Image.new('RGB', (unit_size, target_height))
    left = 0
    right = unit_size
    for image in imagefile:
        target.paste(image, (0, left, unit_size, right))
        left += unit_size
        right += unit_size
    quality_value = 100
    target.save('result.png', quality_value = quality_value)
    print('Done!')


def stiching(images, width, height, unit_size):
    length = len(images)
    imagefile = []
    for img in images:
        imagefile.append(Image.open(img))
    target = Image.new('RGB', (unit_size * width, unit_size * height))
    coor = []
    for i in range(height):
        coor_y = i * unit_size
        coor_height = (i+1) * unit_size
        for j in range(width):
            coor_x = j * unit_size
            coor_width = (j+1) * unit_size
            coor.append((coor_x, coor_y, coor_width, coor_height))
    for i in range(length):
        target.paste(imagefile[i], coor[i])

    quality_value = 100
    target.save('result.png', quality_value = quality_value)
    print('Done!')