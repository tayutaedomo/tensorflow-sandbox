#
# Reference: https://qiita.com/9ryuuuuu/items/e4ee171079ffa4b87424
#

import numpy as np
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Embedding(3, 2))
model.compile(optimizer='rmsprop', loss='mse')
model.summary()

a = np.array([1])
print('input:', a)
print('model.predict:', model.predict(a))

model = keras.Sequential()
model.add(keras.layers.Embedding(3, 2))
model.compile(optimizer='rmsprop', loss='mse')
model.summary()

a = np.array([1])
print('input:', a)
print('model.predict:', model.predict(a))

a = np.array([0, 1, 0, 2])
print('input:', a)
print('model.predict:', model.predict(a))

a = np.array([[0], [1], [0], [2]])
print('input:', a)
print('model.predict:', model.predict(a))

a = np.array([[0, 1], [0, 2]])
print('input:', a)
print('model.predict:', model.predict(a))

