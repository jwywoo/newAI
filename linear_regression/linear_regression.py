# Linear regression
import tensorflow as tf

# tf.enable_eager_execution()

# given x and y
# x -> values creating expected data with W
# y -> actual data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# W -> Weights variable that changes by training that will give us expected value after training
# b -> bias
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# How much result that we will apply to W and b after training
learning_rate = 0.01

# Training
for i in range(100 + 1):
    # Recording operations(Gradient Descent)
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    # getting gradient by tape(Gradient tape)
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:
        print("{:5} | {:10.4f} | {:10.4} | {:10.6f}".format(i, W.numpy(), b.numpy(), cost))
