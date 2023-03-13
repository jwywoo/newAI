import tensorflow as tf
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


# X = [1, 2, 3]
# Y = [1, 2, 3]


def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))


W_values = np.linspace(-3, 5, num=15)
print(W_values)
cost_values = []

for feed_w in W_values:
    curr_cost = cost_func(feed_w, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_w, curr_cost))

# Another version of using tensorflow to implement gradient descent
# tf.set_random_seed(0)

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

# W = tf.Variable(tf.random_normal_initializer([1], -100., 100.))
W = tf.Variable(20.)

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(
            step, cost.numpy(), W.numpy()
        ))
