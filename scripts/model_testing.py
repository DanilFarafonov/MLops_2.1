import pandas as pd
from tensorflow import keras
import json


x_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

model = keras.models.load_model('model')
score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
results = {'loss': score[0], "accuracy": score[1]}
with open("results.json", "w") as file:
    json.dump(results, file)
