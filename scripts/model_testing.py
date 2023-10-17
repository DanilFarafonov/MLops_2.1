import pandas as pd
from tensorflow import keras
import json

# загружаем тестовые данные
x_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# загружаем модель и запускаем тестирование
model = keras.models.load_model('model')
score = model.evaluate(x_test, y_test)

# сохранение результатов эксперимента
results = {'loss': score[0], "accuracy": score[1]}
with open("results.json", "w") as file:
    json.dump(results, file)
