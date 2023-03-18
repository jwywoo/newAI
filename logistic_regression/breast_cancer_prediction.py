import tensorflow as tf
import numpy as np
import pandas as pd

# Prediction for breast cancer
# dataset url: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

# Reading csv using pandas
csv_dataframe = pd.read_csv("breast-cancer.csv")
dataframe_np = csv_dataframe.to_numpy()

# y_values
id_diagnosis = dataframe_np[:, :2]
temp = list()
for i in range(len(id_diagnosis)):
    if id_diagnosis[i][1] == "M":
        temp.append(0)
    else:
        temp.append(1)
diagnosis_values = np.array(temp)
# x_values
symptoms = dataframe_np[:, 2:]

