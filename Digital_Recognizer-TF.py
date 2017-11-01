import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

directory = '../../Datasets/Digital_Recognizer/'
train_input = pd.read_csv(directory + 'train.csv')

train_target = train_input['label'].values
train_input.drop(['label'], axis=1, inplace=True)

train_input = train_input.astype('float32')
train_input = train_input / 255.
train_input = train_input.values

num_classes = 10
image_width = 28
image_height = 28
batch_size = 64
num_epochs = 20
rpt_freq = 50
seed = 7
learning_rate = 1e-4
padding = 'SAME'
keep_prob = 0.95
conv_stride = [1, 1, 1, 1]
pool_stride = [1, 2, 2, 1]
pool_kernel = [1, 2, 2, 1]

train_input = train_input.reshape(-1, image_width, image_height, 1)

f, axarr = plt.subplots(2, 5)
for i in range(0, 2):
    for j in range(0, 5):
        axarr[i][j].imshow(train_input[i * 5 + j, :].reshape(28, 28), cmap=cm.Greys_r)
        axarr[i][j].axis('off')

train_data, validation_data, train_labels, validation_labels = train_test_split(train_input, train_target,
                                                                                test_size=0.20, random_state=0)

train_size = train_data.shape[0]
if train_size%batch_size !=0:
    raise ValueError('Train size is not multiple of batch size')

tf.reset_default_graph()


def error_rate(predictions, labels):
    pred = np.argmax(predictions, 1)
    return 100.0 - (100.0 * np.sum(pred == labels) / predictions.shape[0])


def weight_variable(shape, stddev):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, seed=seed))


def bias_variable(shape, init_value):
    return tf.Variable(tf.constant(init_value, shape=shape))


def model(data, train=False):
    conv = tf.nn.conv2d(data, conv1_weights, strides=conv_stride, padding=padding)
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    conv = tf.nn.conv2d(relu, conv2_weights, strides=conv_stride, padding=padding)
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

    if train:
        relu = tf.nn.dropout(relu, keep_prob)

    pool = tf.nn.max_pool(relu, ksize=pool_kernel, strides=pool_stride, padding=padding)

    conv = tf.nn.conv2d(pool, conv3_weights, strides=conv_stride, padding=padding)
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

    conv = tf.nn.conv2d(relu, conv4_weights, strides=conv_stride, padding=padding)
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))

    if train:
        relu = tf.nn.dropout(relu, keep_prob)

    pool = tf.nn.max_pool(relu, ksize=pool_kernel, strides=pool_stride, padding=padding)

    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    if train:
        hidden = tf.nn.dropout(hidden, keep_prob, seed=seed)

    return tf.matmul(hidden, fc2_weights) + fc2_biases


def eval_in_batches(data, sess):
    size = data.shape[0]
    if size < batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_classes), dtype=np.float32)
    for begin in range(0, size, batch_size):
        end = begin + batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(eval_prediction, feed_dict={x_batch: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(eval_prediction, feed_dict={x_batch: data[-batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


x = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_height, 1], name='X')
y_ = tf.placeholder(tf.int64, shape=[batch_size, ], name='Y')
x_batch = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_height, 1], name='X_batch')

conv1_weights = weight_variable([3, 3, 1, 32], 0.1)
conv1_biases = bias_variable([32], 0.1)

conv2_weights = weight_variable([3, 3, 32, 32], 0.1)
conv2_biases = bias_variable([32], 0.1)

conv3_weights = weight_variable([3, 3, 32, 64], 0.1)
conv3_biases = bias_variable([64], 0.1)

conv4_weights = weight_variable([3, 3, 64, 64], 0.1)
conv4_biases = bias_variable([64], 0.1)

fc1_weights = weight_variable([image_width // 4 * image_height // 4 * 64, 512], 0.1)
fc1_biases = bias_variable([512], 0.1)

fc2_weights = weight_variable([512, num_classes], 0.1)
fc2_biases = bias_variable([num_classes], 0.1)

logits = model(x, True)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_))
train_prediction = tf.nn.softmax(logits)
eval_prediction = tf.nn.softmax(model(x_batch))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

start_time = time.time()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(0, num_epochs):
        for offset in range(0, train_size, batch_size):
            batch_data = train_data[offset:(offset + batch_size), ...]
            batch_labels = train_labels[offset:(offset + batch_size)]

            feed_dict = {x: batch_data, y_: batch_labels}

            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            if offset/batch_size % rpt_freq == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d of epoch %d (%.1f perc.), %.1f ms' %
                      (offset/batch_size, epoch, 100 * offset/train_size,
                       1000 * elapsed_time / rpt_freq))
                print('\tBatch loss: %.3f' % l)
                print('\tBatch error: %.1f%%' % error_rate(predictions, batch_labels))

        print('\tEpoch %d: Validation error: %.1f%%' % (epoch, error_rate(
            eval_in_batches(validation_data, sess), validation_labels)))

    test_error = error_rate(eval_in_batches(validation_data, sess), validation_labels)
    print('Test error: %.1f%%' % test_error)
