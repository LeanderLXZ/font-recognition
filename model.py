import numpy as np
import tensorflow as tf
import pickle


# Hyperparameters

epochs = 10
batch_size = 512
learning_rate = 0.001
keep_prob = 0.5
iteration_print = 10


# Read codes and labels from file

print('Loading codes and labels...')

with open('train_x.p', 'rb') as f:
    train_x = pickle.load(f)

with open('train_y.p', 'rb') as f:
    train_y = pickle.load(f)

with open('val_x.p', 'rb') as f:
    val_x = pickle.load(f)

with open('val_y.p', 'rb') as f:
    val_y = pickle.load(f)

with open('test_x.p', 'rb') as f:
    test_x = pickle.load(f)

with open('test_y.p', 'rb') as f:
    test_y = pickle.load(f)


# Inputs

inputs_ = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]], name='inputs')
labels_ = tf.placeholder(tf.int64, shape=[None, train_y.shape[1]], name='labels')


# Classifier

fc1 = tf.contrib.layers.fully_connected(inputs_,
                                        1024,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        biases_initializer=tf.zeros_initializer())
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

fc2 = tf.contrib.layers.fully_connected(fc1,
                                        512,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        biases_initializer=tf.zeros_initializer())
fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

fc3 = tf.contrib.layers.fully_connected(fc2,
                                        256,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                        biases_initializer=tf.zeros_initializer())
fc3 = tf.nn.dropout(fc3, keep_prob=keep_prob)

logits = tf.contrib.layers.fully_connected(fc3,
                                           train_y.shape[1],
                                           activation_fn=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Operations for validation/test accuracy

predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# Get batches

def get_batches(x, y, n_batches):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x)//n_batches

    for ii in range(0, n_batches*batch_size, batch_size):

        # If not on the last batch, grab data with size batch_size
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size]

        # On the last batch, grab the rest of the data
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y


# Training

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    # TODO: Your training code here
    for i in range(epochs):
        for batch_x, batch_y in get_batches(train_x, train_y, batch_size):
            loss, train_acc, _ = sess.run([cost, accuracy, optimizer],
                                       feed_dict={inputs_: batch_x,
                                                  labels_: batch_y})

            if iteration % iteration_print == 0:
                val_acc = sess.run(accuracy, feed_dict={inputs_: val_x,
                                                        labels_: val_y})

                print('Epochs: {:>3} | Iteration: {:>5} | Loss: {:>9.4f} | Train_acc: {:>6.2f}% | Val_acc: {:.2f}%'
                      .format(i+1, iteration, loss, train_acc * 100, val_acc * 100))

            iteration += 1

    saver.save(sess, "checkpoints/fonts.ckpt")
