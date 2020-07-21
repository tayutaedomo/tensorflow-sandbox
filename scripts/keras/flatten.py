import numpy as np
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Flatten())
model.compile(optimizer='rmsprop', loss='mse')

a = np.array([[0, 1, 2], [3, 4, 5]])
print('input:', a)
print('model.predict:', model.predict(a))

a = np.array([[1, 2, 3, 4]])
print('input:', a)
print('model.predict:', model.predict(a))

a = np.array([1, 2, 3, 4])
print('input:', a)
print('model.predict:', model.predict(a))

a = np.array([[[1, 2], [3, 4]]])
print('input:', a)
print('model.predict:', model.predict(a))

