import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import yaml


# считываем гиперпараметры модели
params = yaml.safe_load(open("scripts/params.yaml"))["model"]

# загружаем тренировочные данные
x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# создание нейронной сети
model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# обучение нейронной сети с указанными гиперпараметрами
model.fit(x_train, y_train,
          batch_size=params["batch_size"],
          epochs=params["epochs"],
          verbose=params["verbose"])

# сохранение модели
model.save('model')
