import tensorflow as tf
import pickle

load_path = './checkpoints/fonts.ckpt'

# Load test codes and labels

print('Loading test codes and labels...')

with open('test_x.p', 'rb') as f:
	test_x = pickle.load(f)

with open('test_y.p', 'rb') as f:
	test_y = pickle.load(f)


loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:

    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    inputs_ = loaded_graph.get_tensor_by_name('inputs:0')
    labels_ = loaded_graph.get_tensor_by_name('labels:0')
    accuracy = loaded_graph.get_tensor_by_name('accuracy:0')

    feed = {inputs_: test_x,
            labels_: test_y}

    test_acc = sess.run(accuracy, feed_dict=feed)

    print("Test accuracy: {:.4f}".format(test_acc * 100))
