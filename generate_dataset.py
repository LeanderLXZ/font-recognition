import os
import numpy as np
import tensorflow as tf
from PIL import Image
from os.path import isdir


source_data_dir = './data/source/'

contents = os.listdir(source_data_dir)
font_classes = [each for each in contents if os.path.isdir(source_data_dir + each)]


# Add Noise

def pre_process_image(image):

    # Randomly crop the input image.
    image = tf.random_crop(image, size=[224, 224, 3])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    # Randomly adjust hue, contrast and saturation.
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    return image


image_ = tf.placeholder(tf.float32, [None, None, None])
image_p_ = pre_process_image(image_)

# Generate Dataset

with tf.Session() as sess:

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
                    img_split = img[y_start:y_end, x_start:x_end, :]

                    #  img = tf.cast(tf.convert_to_tensor(img_split.eval()), tf.float32)

                    img_p = sess.run(image_p_, {image_:img_split})

                    img_s = Image.fromarray(np.uint8(img_p))
                    #  img_s = img_s.resize((224, 224))
                    img_s.save(save_dir + "/{}_{}_{}.jpg".format(e - 30, i, j))
