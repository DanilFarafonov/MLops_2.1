import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=200,
          epochs=10,
          verbose=1)
model.save('model')
