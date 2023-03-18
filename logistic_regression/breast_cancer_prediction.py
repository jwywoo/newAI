import tensorflow as tf
import numpy as np
import pandas as pd

# Prediction for breast cancer
# dataset url: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

# Reading csv using pandas
breast_csv = pd.read_csv("breast-cancer.csv")
breast_np = breast_csv.to_numpy()

# Preparing Values
# y_values
id_diagnosis = breast_np[:, :2]
temp = list()
for i in range(len(id_diagnosis)):
    if id_diagnosis[i][1] == "M":
        temp.append(0)
    else:
        temp.append(1)
diagnosis_values = np.array(temp)
# x_values
symptoms = breast_np[:, 2:]

# Weight and bias
# Dimension of Weight
# number of rows of weight got to be same as number of columns of X
# number of columns of weight got to be same as number of columns of Y
# Current number of columns are 32 thus W matrix got to be 32X1
# print(len(csv_dataframe.columns))
W = tf.Variable(tf.random.normal([len(breast_csv), 1]), name='Weight')
b = tf.Variable(tf.random.normal([1]), name="Bias")


# Hypothesis
def hypoethesis(x):
    return tf.sigmoid(tf.matmul(x, W) + b)

n_epochs = 2000
for i in range(n_epochs):
    with tf.GradientTape as tape:
        # Cost
        cost = tf.reduce_mean(
            diagnosis_values * tf.math.log(hypoethesis(symptoms))
            + (1 - diagnosis_values) * tf.math.log(1 - hypoethesis(symptoms))
        )
