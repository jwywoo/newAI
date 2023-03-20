import tensorflow as tf
import numpy as np

# practice data
# x_data: 8X4
# y_data: 8X3
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# number of columns of y_data
nb_classes = 3

# Weight and Bias
# number of rows of weight got to be same number of column of x_data
# number of column of weight got to be same as y_data's column
W = tf.Variable(tf.random.normal([4, nb_classes]), name="weight")
# bias's rows and columns got to be same as multiplication of x_data and weight
b = tf.Variable(tf.random.normal([nb_classes]), name='bias')


# Hypothesis
# hypothesis = tf.nn.softmax(tf.matmul(x_data,W) + b)
# print(hypothesis)
def hypothesis(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


# hypothesis(x_data)
# sample_db = [[1, 2, 3, 4]]
# sample_db = np.array(sample_db, dtype=np.float32)
# print(hypothesis(sample_db))

# Cost function
def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logits), axis=1)
    return tf.reduce_mean(cost)


def gradient_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X, Y)
        grads = tape.gradient(cost, [W, b])
        return grads


print(gradient_fn(x_data,y_data))
