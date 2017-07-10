import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit


batch_size = 100


data_dir = './data/img/'
contents = os.listdir(data_dir)
font_classes = [font for font in contents if os.path.isdir(data_dir + font)]


codes_list = []
labels = []
batch = []
vgg_codes = None


def pre_process_image(inputs):

    img_batch = tf.Variable(tf.float32, [None, 224, 224, 3])

    for i in range(len(inputs)):
        # Randomly crop the input image.
        image = inputs[i]
        image = tf.random_crop(image, size=[224, 224, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        image_reshaped = tf.reshape(image, [1, 224, 224, 3])

        img_batch = tf.concat(0, [img_batch, image_reshaped])

    return img_batch


with tf.Session() as sess:

    # Build the vgg network
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, None, None, 3])
    img_input_ = pre_process_image(input_)

    with tf.name_scope("content_vgg"):
        vgg.build(img_input_)

    for font in font_classes:
        print("Starting {} images".format(font))
        font_path = data_dir + font
        files = os.listdir(font_path)

        for ii, file in enumerate(files, 1):

            # Add images to the current batch
            img = Image.open(os.path.join(font_path, file))
            img = np.array(img)

            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(font)

            # Running the batch through the network to get the codes
            if ii % batch_size == 0 or ii == len(files):

                # Image batch to pass to VGG network
                images = np.concatenate(batch)

                # Get the values from the relu6 layer of the VGG network
                codes_batch = sess.run(vgg.relu6, feed_dict={input_: images})

                # Building an array of the codes
                if vgg_codes is None:
                    vgg_codes = codes_batch
                else:
                    vgg_codes = np.concatenate((vgg_codes, codes_batch))

                # Reset to start building the next batch
                batch = []
                print('{} images processed'.format(ii))


# Preprocess Data

# One-hot Encode

lb = LabelBinarizer()
lb.fit(labels)

labels_vecs = lb.transform(labels)


# Shuffle and split dataset

ss_train = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
train_idx, test_idx = next(ss_train.split(vgg_codes, labels))

train_x, train_y = vgg_codes[train_idx], labels_vecs[train_idx]
test_x, test_y = vgg_codes[test_idx], labels_vecs[test_idx]

ss_test = StratifiedShuffleSplit(n_splits=10, test_size=0.5)
val_idx, test_idx = next(ss_test.split(test_x, test_y))

val_x, val_y = test_x[val_idx], test_y[val_idx]
test_x, test_y = test_x[test_idx], test_y[test_idx]


# Save data

with open('train_x.p', 'wb') as f:
    pickle.dump(train_x, f)

with open('train_y.p', 'wb') as f:
    pickle.dump(train_y, f)

with open('val_x.p', 'wb') as f:
    pickle.dump(val_x, f)

with open('val_y.p', 'wb') as f:
    pickle.dump(val_y, f)

with open('test_x.p', 'wb') as f:
    pickle.dump(test_x, f)

with open('test_y.p', 'wb') as f:
    pickle.dump(test_y, f)

print('Codes and labels saved.')
