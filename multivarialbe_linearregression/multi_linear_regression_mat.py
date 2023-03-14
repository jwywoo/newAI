import tensorflow as tf
import numpy as np

# 4x3 mat
data = np.array([
    [73., 93., 73., 152.],
    [80., 88., 66., 185.],
    [75., 93., 70., 196.],
    [73., 66., 70., 142.]
], dtype=np.float32)

# x matrix and y matrix
# [:,:] first : -> rows, second : -> columns
# [:,:-1] -> all rows, every column except the last one
# [:,[-1]] -> all rows but only the last column
x = data[:, :-1]
y = data[:, [-1]]

# 3x1 mat
# W = tf.Variable(tf.random.normal([3, 1]))
temp = np.array([
    [5.],
    [5.],
    [5.]
], dtype=np.float32)
W = tf.Variable(temp)
# 1x1 mat
# b = tf.Variable(tf.random.normal([1]))
b = tf.Variable(0.5)
learning_rate = 0.00001


def predict(X):
    return tf.matmul(X, W) + b


n_epochs = 2000
for i in range(n_epochs + 1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(x) - y))

    # W_grad = tape.gradient(cost, W)
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
