# -*- coding: utf-8 -*-

# Sample code to use string producer.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mp

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


num_classes = 3
batch_size = 10


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)

        if i == 0:
            o_h = [1., 0., 0.]
        elif i == 1:
            o_h = [0., 1., 0.]
        else:
            o_h = [0., 0., 1.]

        image, label = tf.image.decode_jpeg(file_image), o_h # one_hot(float(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["data3/train/0/*.jpg", "data3/train/1/*.jpg", "data3/train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data3/valid/0/*.jpg", "data3/valid/1/*.jpg", "data3/valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data3/test/0/*.jpg", "data3/test/1/*.jpg", "data3/test/2/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_valid, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

error_list = []
accuracy_training_list = []

validation_error_list = []
accuracy_validation_list = []

epoch_list = []

threshold = 0.001
patience = 16
patience_count = 0

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(400):

        sess.run(optimizer)

        training_error = sess.run(cost)
        error_list.append(training_error)

        validation_error = sess.run(cost_valid)
        validation_error_list.append(validation_error)

        epoch_list.append(_)

        correct_prediction = tf.equal(tf.argmax(example_batch_train_predicted, 1), tf.argmax(label_batch_train, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accuracy_training_result = sess.run(accuracy)
        accuracy_training_list.append(accuracy_training_result)

        correct_prediction = tf.equal(tf.argmax(example_batch_valid_predicted, 1), tf.argmax(label_batch_valid, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accuracy_validation_result = sess.run(accuracy)
        accuracy_validation_list.append(accuracy_validation_result)

        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            print("Training Error", training_error)
            print("Accuracy Training:" + str(accuracy_training_result))
            print("Validation Error:", validation_error)
            print("Accuracy Validation:" + str(accuracy_validation_result))

        pe = abs(validation_error_list[_] - validation_error_list[_-1])
        if _ > 0 and pe < threshold:
            patience_count += 1
        else:
            patience_count = 0

        if patience_count > patience:
            print("Finalizamos el entrenamiento del modelo de forma temprana.")
            break

    mp.title("Gráfica 1")
    mp.plot(epoch_list, validation_error_list, label='Validation')
    mp.plot(epoch_list, error_list, label='Training')
    mp.xlabel('Número de épocas')
    mp.ylabel('Error')
    mp.legend()
    mp.show()

    mp.title("Gráfica 2")
    mp.plot(epoch_list, accuracy_validation_list, label='Validation')
    mp.plot(epoch_list, accuracy_training_list, label='Training')
    mp.xlabel('Número de épocas')
    mp.ylabel('Exactitud')
    mp.legend()
    mp.show()

    print("---- Testing Model ----")
    correct_prediction = tf.equal(tf.argmax(example_batch_test_predicted, 1), tf.argmax(label_batch_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_testing_result = sess.run(accuracy)
    print ("Accuracy Testing = " + str(accuracy_testing_result))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
