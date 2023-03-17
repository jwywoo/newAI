import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

# X = tf.keras.Input(tf.float32, shape=[None, 2])
# Y = tf.keras.Input(tf.float32, shape=[None, 1])


W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

learning_rate = 0.001


def hypothesis(x):
    return tf.sigmoid(tf.matmul(x, W) + b)

# Before training
test_x = np.array([1,2], dype=np.float32)
temp = tf.sigmoid(tf.matmul)

n_epochs = 2000
for i in range(n_epochs):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(y_data * tf.math.log(hypothesis(x_data)) + (1 - y_data) * tf.math.log(1 - hypothesis(x_data)))

    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

# placeholders for a tensor(To manipulate the data)
# placeholders no longer exist: tf.placeholder -> keras.Input or
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior
# tf.keras.Input( datatype, shape=[how many data are given, dimension of the data])
# None -> not know
# X = tf.keras.Input(tf.float32, shape=[None, 2])
# Y = tf.keras.Input(tf.float32, shape=[None, 1])
#
# # Weight and bias
# # random_normal -> random.normal
# # Dimension of Weight
# # number of rows of weight got to be same as number of columns of X
# # number of columns of weight got to be same as number of columns of Y
# # Dimension of Bias
# # Same as Y
# W = tf.Variable(tf.random.normal([2, 1]), name='weight')
# b = tf.Variable(tf.random.normal([1]), name='bias')
#
#
# # Old style TF.1 -> No more
# # Hypothesis
# # without built in sigmoid
# # hypothesis = tf.math.divide(1., 1. + tf.exp(-1*(tf.matmul(X, W) + b)))
# # built in sigmoid
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# # Cost
# cost = tf.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) * tf.math.log(1 - hypothesis))
# # Gradient descent
