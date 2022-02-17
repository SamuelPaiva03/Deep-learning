import keras
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras import models
from keras import layers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, Nadam
from sklearn.metrics import classification_report, confusion_matrix

nb_classes         = 3
img_rows, img_cols = 48, 48
batch_size         = 32

train_data_dir = ("train")
test_data_dir = ("test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.3,
    horizontal_flip=True
)

train_set = train_datagen.flow_from_directory(
  train_data_dir,
  color_mode  = 'grayscale',
  target_size = (img_rows, img_cols),
  batch_size  = batch_size,
  class_mode  = 'categorical',
  shuffle     = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(
	test_data_dir,
  color_mode  = 'grayscale',
  target_size = (img_rows, img_cols),
  batch_size  = batch_size,
  class_mode  = 'categorical',
  shuffle     = True
)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) #isso converte os mapas de características 3D em vetores de 1D

model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer = Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(x = train_set,epochs = 80,validation_data = test_set)  

y_pred = model.predict_classes(test_set)
class_labels = test_set.class_indices
conf_test = confusion_matrix(test_set.classes, y_pred)
print(conf_test)
fig, ax = plt.subplots(figsize=(4, 4))
ax.matshow(conf_test, cmap=plt.cm.Blues)
print(class_labels)
#ax.set_xticks([0,1,2,3,4])
#ax.set_yticks([0,1,2,3,4])
ax.set_xticklabels([' ', 'feliz','neutro','triste'])
ax.set_yticklabels([' ', 'feliz','neutro','triste'])
  
for i in range(conf_test.shape[0]):
    for j in range(conf_test.shape[1]):
        ax.text(x=j, y=i, s=conf_test[i, j], ha='center')

plt.title("Matriz de confusão")
plt.ylabel("Verdadeiro"), plt.xlabel("Predito")  
plt.show()