import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow_vgg import vgg16
from PIL import Image


batch_size = 100


data_dir = './data/img/'
contents = os.listdir(data_dir)
font_classes = [font for font in contents if os.path.isdir(data_dir + font)]


codes_list = []
labels = []
batch = []
vgg_codes = None


with tf.Session() as sess:

    # Build the vgg network
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])

    with tf.name_scope("content_vgg"):
        vgg.build(input_)

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


# write vgg codes to file
with open('vgg_codes', 'w') as f:
    vgg_codes.tofile(f)


# write labels to file
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
