import cv2
import itertools
import numpy as np


def LogTransform(image_array):
    log = 20 * np.log(image_array * 256 + 1)
    return log

def HistEqualize(image_array):

    equ = cv2.equalizeHist(image_array)
    return equ

def DCT(image_array):
    img = np.float32(image_array) / 255.0
    img_dct = cv2.dct(img)
    img_dct[0][0] = img_dct[0][0] * 1.5

    img_dct= cv2.idct(img_dct)
    dct = np.uint8(img_dct) * 255.0

    return dct

def chunks(l, n):
    """ Yield successive n-sized chunks from l """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def DCT_blocks(image_array):
    if len(image_array.shape) != 2:
        raise ValueError('Input image must be a single channel 2D array')

    height = image_array.shape[0]
    width = image_array.shape[1]

    if(height % 8 != 0) or (width % 8 != 0):
        raise ValueError('Image dimensions (%s, %s) must be multiple of 8' % (height, width))

    img_blocks = [image_array[i : i+8, j: j+8]
                  for (i, j) in itertools.product(range(0, height, 8),
                                                  range(0, width, 8))]
    img_blocks = [np.float32(img) / 255.0 for img in img_blocks]

    dct_blocks = [cv2.dct(img_block) for img_block in img_blocks]

    for dct_block in dct_blocks:
        dct_block[0][0] = dct_block[0][0] * 1.5

    rec_img_blocks = [cv2.idct(dct_block) for dct_block in dct_blocks]
    rec_img_blocks = [np.uint8(rec_img_block) * 255.0 for rec_img_block in rec_img_blocks]

    rec_img = []
    for chunk_row_blocks in chunks(rec_img_blocks, int(width / 8)):
        for row_block_num in range(8):
            for block in chunk_row_blocks:
                rec_img.extend(block[row_block_num])

    rec_img = np.array(rec_img).reshape(height, width)

    return rec_img

def illuminate(image_array):
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    log = LogTransform(image_array)
    equ = HistEqualize(image_array)
    dct = DCT_blocks(image_array)
    final =  0.3 * dct + 0.2 * log + 0.5 * equ
    return final


img = cv2.imread('/home/tony/lvhui/self_face_emotion/images/test5.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (136, 184))
final = HistEqualize(img)
res = np.hstack((img, final))
cv2.imwrite('/home/tony/lvhui/self_face_emotion/images/res4.png', res)