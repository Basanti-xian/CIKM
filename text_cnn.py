import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes_conv1, num_filters_conv1, filter_sizes_conv2, num_filters_conv2, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        # Create a convolution1 + maxpool layer for each filter size
        conv1_pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes_conv1):
            with tf.name_scope("conv1-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters_conv1]
                W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_conv1]), name="b1")
                conv1 = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv1")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                conv1_pooled_outputs.append(pooled)
        # print ("tf.shape(h)", tf.shape(h))
        # print ("tf.shape(conv1_pooled_outputs)",tf.shape(conv1_pooled_outputs))
        # raw_input()

        # Combine all the pooled features
        num_filters_total_conv1 = num_filters_conv1 * len(filter_sizes_conv1)
        self.h_pool1 = tf.concat(conv1_pooled_outputs, 3)
        self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, num_filters_total_conv1])

        '''
        print ("tf.shape(self.h_pool1)",tf.shape(self.h_pool1))
        raw_input()
        self.h_pool1_reshape = tf.reshape(self.h_pool1, [-1, num_filters_total_conv1, 1, num_filters_conv2])
        print ("tf.shape(self.h_pool1_reshape)",tf.shape(self.h_pool1_reshape))
        raw_input()

        # Create a convolution2 + maxpool layer for each filter size
        conv2_pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes_conv2):
            with tf.name_scope("conv2-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, num_filters_total_conv1, 1, num_filters_conv2]
                W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                b2 = tf.Variable(tf.constant(0.1, shape=[num_filters_conv2]), name="b2")
                conv2 = tf.nn.conv2d(
                    self.h_pool1_reshape,
                    W2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv2")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                conv2_pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total_conv2 = num_filters_conv2 * len(filter_sizes_conv2)
        self.h_pool2 = tf.concat(conv2_pooled_outputs, 3)
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, num_filters_total_conv2])
        '''

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool1_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total_conv1, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.normalized_scores = tf.nn.softmax(self.scores, name="normalized_scores")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


