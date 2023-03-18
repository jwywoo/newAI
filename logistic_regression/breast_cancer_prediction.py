import tensorflow as tf
import numpy as np
import pandas as pd


# Prediction for breast cancer
# dataset url: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

# Reading csv using pandas
csv_dataframe = pd.read_csv("breast-cancer.csv")
dataframe_np = csv_dataframe.to_numpy()
# print(dataframe_np)

# y_values
id_diagnosis = dataframe_np[:,:2]
# x_values
symptoms = dataframe_np[:,2:]
