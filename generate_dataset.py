from PIL import Image
from os.path import isdir
import os
import numpy as np


source_data_dir = './data/source/'

contents = os.listdir(source_data_dir)
font_classes = [each for each in contents if os.path.isdir(source_data_dir + each)]


for font_class in font_classes:

    print('Processing {}...'.format(font_class))

    save_dir = './data/img/' + font_class
    if not isdir(save_dir):
        os.mkdir(save_dir)

    font_dir = source_data_dir + font_class
    images = os.listdir(font_dir)

    for e, img_name in list(enumerate(images))[31:231]:

        img_dir = font_dir + '/' + img_name

        # Load image
        img = Image.open(img_dir)
        img = img.convert('RGB')
        img = np.array(img)

        # Split image
        y_start = 0
        y_end = 328

        for i in range(10):
            y_start = y_end + 32
            y_end = y_start + 269
            x_start = 0
            x_end = 127

            for j in range(8):
                x_start = x_end + 7
                x_end = x_start + 269
                img_slipt = img[y_start:y_end, x_start:x_end, :]
                img_s = Image.fromarray(np.uint8(img_slipt))
                #  img_s = img_s.resize((224, 224))
                img_s.save(save_dir + "/{}_{}_{}.jpg".format(e - 30, i, j))
