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


print("Before Training")
prediction = hypothesis(x_data)
print(prediction)
print(tf.argmax(prediction, 1))
print(tf.argmax(y_data, 1))


# hypothesis(x_data)
# sample_db = [[1, 2, 3, 4]]
# sample_db = np.array(sample_db, dtype=np.float32)
# print(hypothesis(sample_db))

# Cost function
# inputs: X data and Y data
# returns: cost of hypothesis(X)
def cost_fn(X, Y):
    logit = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logit), axis=1)
    return tf.reduce_mean(cost)


# Gradient function
# inputs: X data, Y data
# returns: gradients(slope) of current cost
def gradient_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X, Y)
        grads = tape.gradient(cost, [W, b])
        return grads


# Training function
# inputs: X data, Y data, epochs number of training verbose, learning rate
# return: nothing -> training purpose
def fit(X, Y, epochs=2000, verbose=100, learning_rate=0.5):
    for i in range(epochs):
        w_grad, b_grad = gradient_fn(X, Y)
        W.assign_sub(learning_rate * w_grad)
        b.assign_sub(learning_rate * b_grad)
        # Checking whether test is going right or wrong
        if (i == 0) | ((i + 1) % verbose == 0):
            print("Loss at epoch %d, %f" % (i + 1, cost_fn(X, Y).numpy()))


fit(x_data, y_data)

# After training -> argmax
print("After training")
prediction = hypothesis(x_data)
print(prediction)
print(tf.argmax(prediction, 1))
print(tf.argmax(y_data, 1))
