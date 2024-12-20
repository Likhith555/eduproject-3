
# IMPORT LI

import numpy as np
import pandas as pd
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy

# LOADING DATA

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# RESIZE THE DATA TO TRAIN AND TEST

width, height = 28, 28
x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)

#SPLITTING DATA

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# NORMALIZE THE DATA

x_train = (x_train - x_train.mean()) / x_train.std()
x_val = (x_val - x_val.mean()) / x_val.std()
x_test = (x_test - x_test.mean()) / x_test.std()

# ENCODING

num_labels = 10
y_train = to_categorical(y_train, num_labels)
y_val = to_categorical(y_val, num_labels)
y_test = to_categorical(y_test, num_labels)

# MODEL

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))

model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Dense(num_labels, activation='softmax'))

# MODEL COMPILE AND SUMMARY

model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_val, y_val))

score = model.evaluate(x_test, y_test, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

model.summary()
