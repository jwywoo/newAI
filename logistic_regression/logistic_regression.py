import tensorflow as tf
import numpy as np

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# placeholders for a tensor(To manipulate the data)
# placeholders no longer exist: tf.placeholder -> keras.Input or
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior
# tf.keras.Input( datatype, shape=[how many data are given, dimension of the data])
# None -> not know
X = tf.keras.Input(tf.float32, shape=[None, 2])
Y = tf.keras.Input(tf.float32, shape=[None, 1])

# Weight and bias
# random_normal -> random.normal
# Dimension of Weight
# number of rows of weight got to be same as number of columns of X
# number of columns of weight got to be same as number of columns of Y
# Dimension of Bias
# Same as Y
W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


# Old style TF.1 -> No more
# Hypothesis
# without built in sigmoid
# hypothesis = tf.math.divide(1., 1. + tf.exp(-1*(tf.matmul(X, W) + b)))
# built in sigmoid
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# Cost
cost = tf.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) * tf.math.log(1 - hypothesis))
# Gradient descent


