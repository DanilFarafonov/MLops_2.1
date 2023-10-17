import pandas as pd
import tensorflow as tf


# чтение сырых данных
df_x_train = pd.read_csv("data/x_train_raw.csv")
df_y_train = pd.read_csv("data/y_train_raw.csv")
df_x_test = pd.read_csv("data/x_test_raw.csv")
df_y_test = pd.read_csv("data/y_test_raw.csv")


# преобразование pandas.dataframe в numpy.ndarray
x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy()
x_test = df_x_test.to_numpy()
y_test = df_y_test.to_numpy()


# нормализация данных
x_train = x_train / 255
x_test = x_test / 255


# преобразование меток в формат one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# преобразование numpy.ndarray в pandas.dataframe
df_x_train = pd.DataFrame(x_train)
df_y_train = pd.DataFrame(y_train)
df_x_test = pd.DataFrame(x_test)
df_y_test = pd.DataFrame(y_test)


# запись полученных датафреймов в .csv
df_x_train.to_csv("data/x_train.csv", index=False)
df_y_train.to_csv("data/y_train.csv", index=False)
df_x_test.to_csv("data/x_test.csv", index=False)
df_y_test.to_csv("data/y_test.csv", index=False)
