import tensorflow as tf
import numpy as np

x_data = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

# Weight and bias
# Dimension of Weight
# number of rows of weight got to be same as number of columns of X
# number of columns of weight got to be same as number of columns of Y
# Dimension of Bias
# Same as Y
# X = tf.keras.Input(tf.float32, shape=[None, 2])
# Y = tf.keras.Input(tf.float32, shape=[None, 1])
# temp_w = np.array([[.5],[.5]], dtype=np.float32)
W = tf.Variable(tf.random.normal([2, 1]), name='weight')
# W = tf.Variable(temp_w)
# temp_b = np.array([[0.5]], dtype=np.float32)
b = tf.Variable(tf.random.normal([1]), name='bias')
# b = tf.Variable(temp_b)

learning_rate = 0.001


# Hypothesis
def hypothesis(x):
    return tf.sigmoid(tf.matmul(x, W) + b)


print("Before training")
# Before training
test_x = np.array([[1, 2]], dtype=np.float32)
temp = tf.sigmoid(tf.matmul(test_x, W)) + b

if temp > 0.5:
    print("pass")
else:
    print("fail")

n_epochs = 2000
for i in range(n_epochs):
    with tf.GradientTape() as tape:
        # Cost
        cost = tf.reduce_mean(
            y_data * tf.math.log(hypothesis(x_data))
            + (1 - y_data) * tf.math.log(1 - hypothesis(x_data)))

    # Gradient
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

print("After training")
# After training
test_x = np.array([[1, 2]], dtype=np.float32)
temp = tf.sigmoid(tf.matmul(test_x, W)) + b

if temp > 0.5:
    print("pass")
else:
    print("fail")
